"""Train the heart-sound screening model in the on-chip format (tpu_model.py).

    python python/heart_dataset.py         # once: data/heart_windows.pt
    python python/train_heart.py           # writes weights/heart/

Input: 16 frames x 16 filterbank bands (heart_features.py). Output classes
0 = normal, 1 = abnormal (the other 14 outputs are trained towards "never").
A window's class is tpu_model.predicted_class; a recording is abnormal when
the mean of (pooled[1] - pooled[0]) over its windows is positive.

Training: a float stage, then quantisation-aware steps whose forward values
come from the exact integer path (int8 weights << lsh, LUT softmax,
layer_norm_int) with float gradients. Checkpoints are picked by validation
balanced accuracy (the mean of sensitivity and specificity, the PhysioNet
2016 headline metric), and the test split is only scored once at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import tpu_model
from tpu_model import Int8Matrix, TpuModel
from train import int_attention, int_matmul, sat

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "heart_windows.pt"
TOKENS, WIDTH, FEATURES, OUTPUTS, LAYERS = 16, 64, 16, 16, 2
LAYER_NAMES = tpu_model.LAYER_MATRICES
RSQRT = torch.tensor((0,) * 256 + tpu_model.RSQRT_LUT, dtype=torch.int64)   # index = normalised >> 8
POWERS = torch.tensor([1 << b for b in range(48)], dtype=torch.int64)


# Exact integer ops (int64 tensors), one per op in tpu_model.

def int_layer_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.sum(-1, keepdim=True) >> 6
    c = x - mean
    variance = (c * c).sum(-1, keepdim=True)
    bitlen = (variance.unsqueeze(-1) >= POWERS).sum(-1)
    k = (bitlen - 17) >> 1
    zero = torch.zeros_like(k)
    normalised = torch.bitwise_left_shift(
        torch.bitwise_right_shift(variance, torch.where(k >= 0, 2 * k, zero)),
        torch.where(k < 0, -2 * k, zero))
    inv = torch.where(variance > 0, RSQRT[(normalised >> 8).clamp(0, 1023)], torch.zeros_like(variance))
    shift = 13 + k
    rounding = torch.bitwise_left_shift(torch.ones_like(shift), shift - 1)
    return sat(torch.bitwise_right_shift(c * inv + rounding, shift))


def int_forward(model: TpuModel, features: torch.Tensor) -> torch.Tensor:
    """Pooled logits [batch, 16] of tpu_model.forward, batched."""
    q = lambda m: torch.tensor(m.q8_8(), dtype=torch.int64)
    x = sat(int_matmul(features.long(), q(model.embed)) + torch.tensor(model.position))
    for weights in model.layers:
        w = {name: q(weights[name]) for name in LAYER_NAMES}
        r = sat(x + int_attention(x, w))
        n = int_layer_norm(r)
        h = int_matmul(n, w["w1"]).clamp(min=0)
        x = sat(int_matmul(h, w["w2"]) + n)
    return int_matmul(x, q(model.output)).sum(1)


def quantize_tensor(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """tpu_model.quantize on a float tensor: (int8 values, lsh)."""
    peak = (w.detach().abs().max() * 256).item()
    lsh = 0
    while lsh < tpu_model.MAX_LSH and peak / (1 << lsh) > 127.5:
        lsh += 1
    return torch.round(w.detach() * 256 / (1 << lsh)).clamp(-128, 127), lsh


def ste(real: torch.Tensor, integer: torch.Tensor) -> torch.Tensor:
    return real + (integer.double() / 256 - real).detach()


def to_int(x: torch.Tensor) -> torch.Tensor:
    return sat(torch.round(x.detach() * 256)).long()


class HeartNet(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def weight(rows: int, cols: int, scale: float = 1.0) -> torch.nn.Parameter:
            return torch.nn.Parameter(torch.randn(rows, cols, generator=g, dtype=torch.float64)
                                      * scale / rows ** 0.5)

        self.w = torch.nn.ParameterDict({"embed": weight(FEATURES, WIDTH, 0.5)})
        for layer in range(LAYERS):
            for name in LAYER_NAMES:
                self.w[f"l{layer}_{name}"] = weight(WIDTH, WIDTH)
        self.w["out"] = weight(WIDTH, OUTPUTS, 0.25)
        position = torch.tensor(tpu_model.sinusoidal_position(), dtype=torch.float64) / 256
        self.position = torch.nn.Parameter(position * 0.5)

    def weight(self, key: str, quant: bool) -> torch.Tensor:
        w = self.w[key]
        if not quant:
            return w
        values, lsh = quantize_tensor(w)
        return w + (values * (1 << lsh) / 256 - w).detach()

    def forward(self, features: torch.Tensor, quant: bool) -> torch.Tensor:
        """Pooled logits in real units, [batch, 16]."""
        f = features.double() / 256
        emb = self.weight("embed", quant)
        pos = self.position if not quant else self.position + (to_int(self.position).double() / 256
                                                                 - self.position).detach()
        x = f @ emb + pos
        if quant:
            x = ste(x, sat(int_matmul(features.long(), to_int(emb)) + to_int(pos)))
        for layer in range(LAYERS):
            w = {name: self.weight(f"l{layer}_{name}", quant) for name in LAYER_NAMES}
            q, k, v = x @ w["q"], x @ w["k"], x @ w["v"]
            attention = torch.softmax(q @ k.transpose(-1, -2) / 8, -1) @ v @ w["o"]
            if quant:
                attention = ste(attention, int_attention(to_int(x), {n: to_int(w[n]) for n in "qkvo"}))
            residual = x + attention
            normalised = F.layer_norm(residual, (WIDTH,), eps=1e-5)
            if quant:
                normalised = ste(normalised, int_layer_norm(sat(to_int(residual))))
            hidden = torch.relu(normalised @ w["w1"])
            if quant:
                hidden = ste(hidden, int_matmul(to_int(normalised), to_int(w["w1"])).clamp(min=0))
            out = hidden @ w["w2"]
            if quant:
                out = ste(out, int_matmul(to_int(hidden), to_int(w["w2"])))
            x = out + normalised
        out_w = self.weight("out", quant)
        logits = x @ out_w
        if quant:
            logits = ste(logits, int_matmul(to_int(x), to_int(out_w)))
        return logits.sum(1)

    def export(self) -> TpuModel:
        def matrix(key: str) -> Int8Matrix:
            values, lsh = quantize_tensor(self.w[key])
            return Int8Matrix(tuple(tuple(int(v) for v in row) for row in values.long().tolist()), lsh)
        return TpuModel(
            embed=matrix("embed"),
            layers=tuple({name: matrix(f"l{layer}_{name}") for name in LAYER_NAMES} for layer in range(LAYERS)),
            output=matrix("out"),
            position=tuple(tuple(int(v) for v in row) for row in to_int(self.position).tolist()),
        )


# Data and metrics.

class Data:
    def __init__(self, cache: Path):
        blob = torch.load(cache)
        self.features = blob["features"]
        self.recording = blob["recording"].long()
        self.labels = blob["labels"]
        self.databases = blob["databases"]
        self.splits = blob["splits"]
        self.window_label = self.labels[self.recording]
        split_of = torch.tensor([("train", "val", "test").index(s) for s in self.splits])
        self.window_split = split_of[self.recording]

    def windows(self, split: str) -> torch.Tensor:
        return torch.nonzero(self.window_split == ("train", "val", "test").index(split)).flatten()


def recording_metrics(data: Data, margins: torch.Tensor, index: torch.Tensor) -> dict:
    """Per-recording decisions from window margins (pooled[1] - pooled[0])."""
    recs = data.recording[index]
    totals = torch.zeros(len(data.labels), dtype=torch.float64).index_add_(0, recs, margins.double())
    counts = torch.zeros(len(data.labels)).index_add_(0, recs, torch.ones(len(recs)))
    chosen = counts > 0
    abnormal = totals[chosen] > 0
    truth = data.labels[chosen] == 1
    dbs = [d for d, keep in zip(data.databases, chosen.tolist()) if keep]

    def summary(mask: torch.Tensor) -> dict:
        tp = (abnormal & truth & mask).sum().item(); fn = (~abnormal & truth & mask).sum().item()
        tn = (~abnormal & ~truth & mask).sum().item(); fp = (abnormal & ~truth & mask).sum().item()
        se = tp / (tp + fn) if tp + fn else float("nan")
        sp = tn / (tn + fp) if tn + fp else float("nan")
        return {"recordings": int(mask.sum()), "sensitivity": se, "specificity": sp,
                "balanced_accuracy": (se + sp) / 2, "accuracy": (tp + tn) / max(1, int(mask.sum()))}

    out = summary(torch.ones_like(truth))
    out["per_database"] = {db: summary(torch.tensor([d == db for d in dbs])) for db in sorted(set(dbs))}
    # The pooled number is dominated by the largest database (e); the mean over
    # databases that have both classes is the honest one for unseen hardware.
    scored = [m["balanced_accuracy"] for m in out["per_database"].values()
              if m["balanced_accuracy"] == m["balanced_accuracy"]]
    out["macro_balanced_accuracy"] = sum(scored) / len(scored)
    return out


def margins_of(model: TpuModel, data: Data, index: torch.Tensor, batch: int = 128) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for start in range(0, len(index), batch):
            pooled = int_forward(model, data.features[index[start:start + batch]])
            out.append(pooled[:, 1] - pooled[:, 0])
    return torch.cat(out)


def report(name: str, m: dict) -> None:
    print(f"{name}: {m['recordings']} recordings, sensitivity {m['sensitivity']:.1%}, "
          f"specificity {m['specificity']:.1%}, balanced {m['balanced_accuracy']:.1%}, "
          f"mean over databases {m['macro_balanced_accuracy']:.1%}")
    for db, d in m.get("per_database", {}).items():
        print(f"   db {db}: n={d['recordings']:4d} Se {d['sensitivity']:.1%} Sp {d['specificity']:.1%}")


def train(args) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = Data(args.cache)
    train_idx, val_idx, test_idx = (data.windows(s) for s in ("train", "val", "test"))
    # Sample every (database, label) group equally, so a recording's source
    # carries no information about its label during training.
    window_db = [data.databases[r] for r in data.recording[train_idx].tolist()]
    groups: dict[tuple[str, int], list[int]] = {}
    for w, db, label in zip(train_idx.tolist(), window_db, data.window_label[train_idx].tolist()):
        groups.setdefault((db, label), []).append(w)
    groups = {key: idx for key, idx in groups.items() if len(idx) >= 32}
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}; "
          f"sampling {len(groups)} (database, label) groups equally")
    net = HeartNet(args.seed)
    best = (-1.0, None, None)
    for stage, steps, lr, quant in (("float", args.steps, 2e-3, False), ("qat", args.qat_steps, 3e-4, True)):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        start = time.perf_counter()
        for step in range(1, steps + 1):
            keys = [rng.choice(sorted(groups)) for _ in range(args.batch)]
            index = torch.tensor([rng.choice(groups[key]) for key in keys])
            loss = F.cross_entropy(net(data.features[index], quant), data.window_label[index])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.eval_every == 0 or step == steps:
                model = net.export()
                metrics = recording_metrics(data, margins_of(model, data, val_idx), val_idx)
                print(f"{stage} {step}: loss {loss.item():.3f}, val mean-over-databases "
                      f"{metrics['macro_balanced_accuracy']:.1%}, pooled {metrics['balanced_accuracy']:.1%} "
                      f"(Se {metrics['sensitivity']:.1%}, Sp {metrics['specificity']:.1%}), "
                      f"{time.perf_counter() - start:.0f} s", flush=True)
                if quant and metrics["macro_balanced_accuracy"] > best[0]:
                    best = (metrics["macro_balanced_accuracy"], model, metrics)
    model, val_metrics = best[1], best[2]
    test_metrics = recording_metrics(data, margins_of(model, data, test_idx), test_idx)
    report("validation (best checkpoint)", val_metrics)
    report("test", test_metrics)
    # The batched integer path must equal the spec the FPGA implements.
    sample = data.features[test_idx[:4]]
    exact = [tpu_model.forward(model, row.tolist()) for row in sample]
    batched = int_forward(model, sample)
    agree = all([sum(r[j] for r in logits) for j in range(OUTPUTS)] == batched[i].tolist()
                for i, logits in enumerate(exact))
    print("batched integer path == tpu_model.forward:", agree)
    tpu_model.export(args.output, model, {"task": "heart-sound screening, 0 normal / 1 abnormal",
                                          "features": "heart_features.py"})
    (args.output / "metrics.json").write_text(json.dumps(
        {"validation": val_metrics, "test": test_metrics, "exact_check": agree, "args": vars(args)},
        indent=2, default=str) + "\n")
    print(f"exported to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "weights" / "heart")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--qat-steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
