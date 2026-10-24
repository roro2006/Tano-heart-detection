"""Train the two-layer pattern denoiser and export it in Q8.8.

    python python/train.py [--output weights] [--steps 3000] [--qat-steps 1500]

Target: reconstruct the clean shifted pattern (dataset.py ``target``) from a
noisy grid, which is what app.py draws. The model is exactly the one
model_export.full_inference runs: embed_grid, then per layer attention_q8_8
(Wq/Wk/Wv/Wo, on the FPGA) and finish_layer (residual, layer norm, ReLU FFN
w1/w2), then output_projection (64x16, sigmoid).

Training runs in two stages. The float stage mirrors those ops in float64.
The quantization-aware stage keeps the float ops for gradients but replaces
every forward value with the integer path's (Q8.8 weights, floor-shifted
products, EXP_LUT softmax, Q8.8 rounding), so the loss sees what the FPGA
computes. Accuracy is then measured with full_inference itself, not with
this module (see evaluate.py).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise RuntimeError("install torch to train the model") from exc

sys.path.insert(0, str(Path(__file__).parent))
import dataset
import tpu_model
from evaluate import binarize, held_out, report, score
from model_export import QuantModel, export_model, positional_embedding
from q8_attention import EXP_LUT

ROOT = Path(__file__).resolve().parents[1]
TOKENS, WIDTH, PIXELS, LAYERS = 16, 64, 16, 2
ATTENTION = ("q", "k", "v", "o")
I16_MIN, I16_MAX = -(1 << 15), (1 << 15) - 1
LUT = torch.tensor(EXP_LUT, dtype=torch.int64)
POSITION = torch.tensor(positional_embedding(TOKENS, WIDTH), dtype=torch.int64)


# Integer path on int64 tensors, one function per op in model_export/q8_attention.

def sat(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(I16_MIN, I16_MAX)


def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return sat(((a.unsqueeze(-1) * b) >> 8).sum(-2))


def int_softmax(scores: torch.Tensor) -> torch.Tensor:
    maximum = scores.max(-1, keepdim=True).values
    exps = LUT[((maximum - scores) >> 4).clamp(0, 255)]
    probs = (exps << 16) // exps.sum(-1, keepdim=True)
    remainder = (1 << 16) - probs.sum(-1, keepdim=True)
    return probs.scatter_add(-1, scores.argmax(-1, keepdim=True), remainder)


def int_attention(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    q, k, v = (int_matmul(x, w[name]) for name in "qkv")
    scores = sat(((q.unsqueeze(-2) * k.unsqueeze(-3)) >> 8).sum(-1) >> 3)
    probs = int_softmax(scores)
    attended = sat((probs.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2) >> 16)
    return int_matmul(attended, w["o"])


def int_layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    real = x.double()
    mean = real.mean(-1, keepdim=True)
    variance = ((real - mean) ** 2).mean(-1, keepdim=True) / 65536.0
    scale = 1.0 / (variance + eps) ** 0.5
    return sat(torch.round(((real - mean) / 256.0) * scale * 256)).long()


def int_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return sat(torch.round(256 / (1 + torch.exp(-x.double() / 256)))).long()


def int_forward(pixels: torch.Tensor, model: QuantModel) -> torch.Tensor:
    """Batched equivalent of full_inference with the software oracle."""
    x = sat(F.pad(pixels.long() * 256, (0, WIDTH - PIXELS)) + POSITION)
    for layer in range(LAYERS):
        w = {name: torch.tensor(model.attention[layer][name]) for name in ATTENTION}
        w1, w2 = (torch.tensor(m) for m in model.feed_forward[layer])
        normalized = int_layer_norm(sat(x + int_attention(x, w)))
        hidden = int_matmul(normalized, w1).clamp(min=0)
        x = sat(int_matmul(hidden, w2) + normalized)
    return int_sigmoid(int_matmul(x, torch.tensor(model.output)))


# Float model. In QAT mode each block returns the integer value with the float gradient.

def to_int(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.detach() * 256).long()


def ste(real: torch.Tensor, integer: torch.Tensor) -> torch.Tensor:
    return real + (integer.double() / 256 - real).detach()


class Denoiser(torch.nn.Module):
    def __init__(self, hidden: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def weight(rows: int, cols: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(torch.randn(rows, cols, generator=g, dtype=torch.float64) / rows ** 0.5)

        self.w = torch.nn.ParameterDict()
        for layer in range(LAYERS):
            for name in ATTENTION:
                self.w[f"l{layer}_{name}"] = weight(WIDTH, WIDTH)
            self.w[f"l{layer}_w1"] = weight(WIDTH, hidden)
            self.w[f"l{layer}_w2"] = weight(hidden, WIDTH)
        self.w["out"] = weight(WIDTH, PIXELS)

    def weight(self, key: str, quant: bool) -> torch.Tensor:
        w = self.w[key]
        if not quant:
            return w
        return w + (torch.round(w * 256).clamp(I16_MIN, I16_MAX) / 256 - w).detach()

    def forward(self, pixels: torch.Tensor, quant: bool) -> torch.Tensor:
        """Return output logits (pre-sigmoid, real units) of shape [batch, 16, 16]."""
        x = F.pad(pixels.double(), (0, WIDTH - PIXELS)) + POSITION.double() / 256
        for layer in range(LAYERS):
            w = {name: self.weight(f"l{layer}_{name}", quant) for name in ATTENTION}
            q, k, v = x @ w["q"], x @ w["k"], x @ w["v"]
            attention = torch.softmax(q @ k.transpose(-1, -2) / 8, -1) @ v @ w["o"]
            if quant:
                attention = ste(attention, int_attention(to_int(x), {n: to_int(t) for n, t in w.items()}))
            residual = x + attention
            normalized = F.layer_norm(residual, (WIDTH,), eps=1e-5)
            if quant:
                normalized = ste(normalized, int_layer_norm(sat(to_int(residual))))
            w1, w2 = self.weight(f"l{layer}_w1", quant), self.weight(f"l{layer}_w2", quant)
            hidden = torch.relu(normalized @ w1)
            if quant:
                hidden = ste(hidden, int_matmul(to_int(normalized), to_int(w1)).clamp(min=0))
            out = hidden @ w2
            if quant:
                out = ste(out, int_matmul(to_int(hidden), to_int(w2)))
            x = out + normalized
        wout = self.weight("out", quant)
        logits = x @ wout
        return ste(logits, int_matmul(to_int(x), to_int(wout))) if quant else logits

    def quantized(self) -> QuantModel:
        def q(key: str) -> list[list[int]]:
            return torch.round(self.w[key].detach() * 256).clamp(I16_MIN, I16_MAX).long().tolist()
        return QuantModel(
            attention=[{name: q(f"l{layer}_{name}") for name in ATTENTION} for layer in range(LAYERS)],
            feed_forward=[(q(f"l{layer}_w1"), q(f"l{layer}_w2")) for layer in range(LAYERS)],
            output=q("out"),
        )


# Data: the dataset.py training split plus fresh, noisier corruptions of the same patterns.

def augmented(rng: random.Random, count: int) -> list[dict]:
    base = dataset.patterns()
    names = list(base)
    records = []
    for _ in range(count):
        target = dataset.shifted(base[rng.choice(names)], rng.randint(-2, 2), rng.randint(-2, 2))
        pixels = dataset.corrupt(rng, target, drop=rng.uniform(0, 0.25), salt=rng.uniform(0, 0.04))
        records.append({"pixels": pixels, "target": target})
    return records


def batch(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor([r["pixels"] for r in records], dtype=torch.float64),
            torch.tensor([r["target"] for r in records], dtype=torch.float64))


def train_stage(model: Denoiser, train: list[dict], rng: random.Random, steps: int,
                lr: float, quant: bool, batch_size: int, check: list[dict]) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        half = batch_size // 2
        pixels, target = batch(rng.sample(train, half) + augmented(rng, batch_size - half))
        loss = F.binary_cross_entropy_with_logits(model(pixels, quant), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step == steps:
            print(f"{'qat' if quant else 'float'} step {step}: loss {loss.item():.4f}, "
                  f"held-out pixel {quick_accuracy(model, check):.2%} (integer path), "
                  f"{time.perf_counter() - start:.0f} s", flush=True)


def predict(model: QuantModel, records: list[dict]) -> list[list[list[int]]]:
    with torch.no_grad():
        return [binarize(o) for o in int_forward(batch(records)[0], model).tolist()]


def quick_accuracy(model: Denoiser, records: list[dict]) -> float:
    return score(records, predict(model.quantized(), records))["pixel_accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "weights")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "patterns.json")
    parser.add_argument("--steps", type=int, default=3000, help="float training steps")
    parser.add_argument("--qat-steps", type=int, default=1500, help="quantization-aware steps")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64, help="feed-forward width")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if not args.data.exists():
        dataset.build(args.data)
    train, _ = dataset.split(dataset.load(args.data))
    sets = held_out(args.data)
    model = Denoiser(args.hidden, args.seed)
    train_stage(model, train, rng, args.steps, 3e-3, False, args.batch, sets["test"])
    train_stage(model, train, rng, args.qat_steps, 5e-4, True, args.batch, sets["test"])

    quantized = model.quantized()
    metrics = {}
    for name, records in sets.items():
        metrics[name] = score(records, predict(quantized, records))
        report(f"{name} (integer path)", metrics[name])
    export_model(args.output, quantized, {
        "task": "reconstruct the clean shifted pattern from a noisy 16x16 grid",
        "threshold": 128,
        "hidden": args.hidden,
        "train": {"steps": args.steps, "qat_steps": args.qat_steps, "batch": args.batch, "seed": args.seed},
        "integer_path_metrics": metrics,
    })
    # The fully on-chip model (int8 weights, integer layer norm); see tpu_model.py.
    tpu_model.export(args.output, tpu_model.from_quant_model(quantized),
                     {"source": "train.py via tpu_model.from_quant_model"})
    print(f"exported to {args.output}")


if __name__ == "__main__":
    main()
