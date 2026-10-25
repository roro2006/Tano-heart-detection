"""Train and evaluate AF-versus-non-AF on 16-beat ECG sequences (ecg_dataset.py).

    python python/ecg_dataset.py                    # once: data/ecg_af_sequences.npz
    python python/train_ecg.py --fold 0             # one cross-validation fold (0-4)
    python python/train_ecg.py --summary            # pool the fold results
    python python/train_ecg.py --final              # train the deployable model -> weights/ecg_af/

Evaluation is patient-grouped 5-fold cross-validation: each AFDB record is
one subject, and every sequence of a record stays in one fold. For test fold
k, fold k+1 is the validation fold (normalisation is fitted on the training
folds only; checkpoints and the decision threshold are chosen on the
validation fold only) and the other three train.

Two models per fold:
- baseline: logistic regression on four interval statistics of the 16 RR
  intervals (coefficient of variation, RMSSD / mean RR, mean |dRR| / mean RR,
  fraction of |dRR| > 50 ms); this measures what rhythm irregularity alone gives.
- transformer: the on-chip model (tpu_model.py, the same HeartNet as
  train_heart.py): 16 beats x 16 features in Q8.8, float stage then
  quantisation-aware steps on the exact integer path. Its score is
  pooled[1] - pooled[0]; AF when it exceeds the validation-chosen threshold.
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
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ecg_dataset
import tpu_model
from train_heart import HeartNet, int_forward

ROOT = Path(__file__).resolve().parents[1]
FOLDS = 5
CLIP = 6.0            # normalised features are clipped to +-6 before Q8.8 (+-1536)
OUT = ROOT / "experiments" / "ecg_af"
# "rhythm": interval features + template consistency + missing fraction, no waveform shape.
FEATURE_SETS = {"all": tuple(range(16)), "rhythm": (0, 1, 2, 3, 14, 15)}


# Data, folds and normalisation.

class Data:
    def __init__(self, cache: Path = ecg_dataset.CACHE):
        blob = np.load(cache)
        self.X, self.rr, self.y = blob["X"], blob["rr"], blob["y"].astype(np.int64)
        self.record, self.records = blob["record"].astype(np.int64), [str(r) for r in blob["records"]]
        self.features = [str(f) for f in blob["features"]]


def fold_of_records(data: Data, seed: int = 2017) -> dict[str, int]:
    """Record -> fold, dealing records sorted by AF share round-robin so every fold mixes AF burdens."""
    share = [(float((data.y[data.record == i] == 1).mean()), name) for i, name in enumerate(data.records)]
    rng = random.Random(seed)
    order = sorted(share)
    out: dict[str, int] = {}
    for start in range(0, len(order), FOLDS):
        block = [name for _, name in order[start:start + FOLDS]]
        slots = list(range(FOLDS))
        rng.shuffle(slots)
        out.update(zip(block, slots))
    return out


def split(data: Data, test_fold: int | None, folds: dict[str, int]) -> dict[str, list[str]]:
    """Records per role. test_fold None (the final model): fold 0 validates, the rest train."""
    val_fold = 0 if test_fold is None else (test_fold + 1) % FOLDS
    roles = {"train": [], "val": [], "test": []}
    for name, f in sorted(folds.items()):
        roles["test" if f == test_fold else "val" if f == val_fold else "train"].append(name)
    return roles


def index_of(data: Data, names: list[str]) -> np.ndarray:
    ids = [data.records.index(n) for n in names]
    return np.flatnonzero(np.isin(data.record, ids))


def fit_normaliser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature median and robust scale (1.4826 x MAD) over training beats."""
    flat = X.reshape(-1, X.shape[-1])
    centre = np.median(flat, axis=0)
    scale = 1.4826 * np.median(np.abs(flat - centre), axis=0)
    return centre, np.where(scale > 1e-9, scale, 1.0)


def quantise(X: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """Float features -> the core's Q8.8 integer tokens."""
    z = np.clip((X - centre) / scale, -CLIP, CLIP)
    return torch.from_numpy(np.round(z * 256).astype(np.int64))


# Metrics.

def metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pred = scores > threshold
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    return {"n": int(len(y)), "af": int((y == 1).sum()), "sensitivity": se, "specificity": sp,
            "balanced_accuracy": np.nanmean([se, sp]), "precision": ppv,
            "average_precision": average_precision(scores, y)}


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn computes it)."""
    if not (y == 1).any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = (y[order] == 1).astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (used on validation patients only)."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, 199)))
    best = max(candidates, key=lambda t: metrics(scores, y, t)["balanced_accuracy"])
    return float(best)


def per_record(data: Data, idx: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    out = {}
    for rec in np.unique(data.record[idx]):
        m = data.record[idx] == rec
        out[data.records[rec]] = metrics(scores[m], data.y[idx][m], threshold)
    return out


# Baseline: interval statistics + logistic regression.

def interval_stats(rr: np.ndarray) -> np.ndarray:
    mean = rr.mean(axis=1)
    d = np.diff(rr, axis=1)
    return np.stack([rr.std(axis=1) / mean, np.sqrt((d ** 2).mean(axis=1)) / mean,
                     np.abs(d).mean(axis=1) / mean, (np.abs(d) > 0.05).mean(axis=1)], axis=1)


def baseline(data: Data, roles: dict) -> dict:
    tr, va, te = (index_of(data, roles[k]) for k in ("train", "val", "test"))
    S = interval_stats(data.rr)
    mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-9
    Z = torch.tensor((S - mu) / sd, dtype=torch.float64)
    y = torch.tensor(data.y, dtype=torch.float64)
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pos = float(y[tr].mean())
    weights = torch.where(y[tr] == 1, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z[tr] @ w + b, y[tr], weight=weights)
        loss.backward()
        return loss
    opt.step(closure)
    scores = (Z @ w + b).detach().numpy()
    threshold = best_threshold(scores[va], data.y[va])
    result = {"weights": w.detach().tolist(), "bias": float(b), "mean": mu.tolist(), "std": sd.tolist(),
              "threshold": threshold, "val": metrics(scores[va], data.y[va], threshold)}
    if len(te):
        result["test"] = metrics(scores[te], data.y[te], threshold)
        result["test_per_record"] = per_record(data, te, scores[te], threshold)
    return result


# Transformer.

def margins(model: tpu_model.TpuModel, Q: torch.Tensor, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(Q), batch):
            pooled = int_forward(model, Q[s:s + batch])
            out.append((pooled[:, 1] - pooled[:, 0]).numpy())
    return np.concatenate(out)


def transformer(data: Data, roles: dict, args) -> tuple[dict, tpu_model.TpuModel, dict]:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tr, va, te = (index_of(data, roles[k]) for k in ("train", "val", "test"))
    centre, scale = fit_normaliser(data.X[tr])
    Q = quantise(data.X, centre, scale)
    # A feature set leaves the other columns at zero (the token layout stays 16 x 16).
    zeroed = [j for j in range(Q.shape[-1]) if j not in FEATURE_SETS[args.feature_set]]
    Q[:, :, zeroed] = 0
    y = torch.from_numpy(data.y)
    # Every (record, label) group is sampled equally: no patient or class dominates.
    groups: dict[tuple[int, int], list[int]] = {}
    for i in tr.tolist():
        groups.setdefault((int(data.record[i]), int(data.y[i])), []).append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= 16}
    keys = sorted(groups)
    va_eval = va[::args.val_stride]
    net = HeartNet(args.seed)
    best = (-1.0, None, None)
    for stage, steps, lr, quant in (("float", args.steps, 2e-3, False), ("qat", args.qat_steps, 3e-4, True)):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        start = time.perf_counter()
        for step in range(1, steps + 1):
            index = torch.tensor([rng.choice(groups[rng.choice(keys)]) for _ in range(args.batch)])
            loss = F.cross_entropy(net(Q[index], quant), y[index])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.eval_every == 0 or step == steps:
                model = net.export()
                s = margins(model, Q[va_eval])
                t = best_threshold(s, data.y[va_eval])
                m = metrics(s, data.y[va_eval], t)
                print(f"{stage} {step}: loss {loss.item():.3f}, val balanced {m['balanced_accuracy']:.1%} "
                      f"(Se {m['sensitivity']:.1%}, Sp {m['specificity']:.1%}, AP {m['average_precision']:.3f}), "
                      f"{time.perf_counter() - start:.0f} s", flush=True)
                if quant and m["balanced_accuracy"] > best[0]:
                    best = (m["balanced_accuracy"], model, t)
    model, _ = best[1], None
    s_va = margins(model, Q[va])
    threshold = best_threshold(s_va, data.y[va])
    result = {"threshold": threshold, "val": metrics(s_va, data.y[va], threshold)}
    if len(te):
        s_te = margins(model, Q[te])
        result["test"] = metrics(s_te, data.y[te], threshold)
        result["test_per_record"] = per_record(data, te, s_te, threshold)
    # The batched integer path must equal the spec the FPGA implements.
    sample = Q[va[:4]]
    exact = [tpu_model.forward(model, row.tolist()) for row in sample]
    batched = int_forward(model, sample)
    result["exact_check"] = all([sum(r[j] for r in lg) for j in range(16)] == batched[i].tolist()
                                for i, lg in enumerate(exact))
    norm = {"centre": centre.tolist(), "scale": scale.tolist(), "clip": CLIP, "zeroed": zeroed,
            "q": "round(clip((x - centre) / scale) * 256), then the 'zeroed' feature columns set to 0"}
    return result, model, norm


def fmt(m: dict) -> str:
    return (f"Se {m['sensitivity']:.1%}, Sp {m['specificity']:.1%}, balanced {m['balanced_accuracy']:.1%}, "
            f"AP {m['average_precision']:.3f} (n={m['n']}, AF {m['af']})")


def run_fold(args) -> None:
    torch.set_num_threads(args.threads)
    data = Data()
    folds = fold_of_records(data)
    roles = split(data, args.fold, folds)
    print(f"fold {args.fold}: train {roles['train']}, val {roles['val']}, test {roles['test']}", flush=True)
    base = baseline(data, roles)
    print("baseline test:", fmt(base["test"]), flush=True)
    trans, _, norm = transformer(data, roles, args)
    print("transformer test:", fmt(trans["test"]), "| exact:", trans["exact_check"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold{args.fold}.json").write_text(json.dumps(
        {"fold": args.fold, "roles": roles, "baseline": base, "transformer": trans, "normaliser": norm,
         "args": vars(args)}, indent=2, default=lambda o: str(o) if isinstance(o, Path) else float(o)) + "\n")


def summary() -> None:
    rows = [json.loads((OUT / f"fold{k}.json").read_text()) for k in range(FOLDS) if (OUT / f"fold{k}.json").exists()]
    print(f"{len(rows)} folds")
    for name in ("baseline", "transformer"):
        per_rec = {}
        for r in rows:
            per_rec.update(r[name]["test_per_record"])
        keys = ("sensitivity", "specificity", "balanced_accuracy", "average_precision")
        fold_means = {k: np.nanmean([r[name]["test"][k] for r in rows]) for k in keys}
        print(f"{name}: mean over folds " + ", ".join(f"{k} {v:.1%}" if k != "average_precision" else f"AP {v:.3f}"
                                                      for k, v in fold_means.items()))
        for rec, m in sorted(per_rec.items()):
            print(f"   {rec}: n={m['n']:5d} AF {m['af']:5d}  Se {m['sensitivity']:.1%}  Sp {m['specificity']:.1%}")
    out = {name: {"folds": [r[name]["test"] for r in rows],
                  "per_record": {k: v for r in rows for k, v in r[name]["test_per_record"].items()}}
           for name in ("baseline", "transformer")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=lambda o: str(o) if isinstance(o, Path) else float(o)) + "\n")


def final(args) -> None:
    """Deployable model: folds 1-4 train, fold 0 validates (checkpoint + threshold). No test fold:
    the cross-validation above is the estimate of how it does on new patients."""
    torch.set_num_threads(args.threads)
    data = Data()
    folds = fold_of_records(data)
    roles = split(data, None, folds)
    trans, model, norm = transformer(data, roles, args)
    base = baseline(data, roles)
    manifest = json.loads(ecg_dataset.MANIFEST.read_text())
    tpu_model.export(args.output, model, {"task": "ECG atrial fibrillation, 0 non-AF / 1 AF",
                                          "features": "ecg_dataset.py"})
    config = {
        "task": "AF versus non-AF on 16 consecutive beats of single-lead ECG",
        "classes": {"0": "non-AF (all other annotated rhythms except flutter and junctional)", "1": "AF"},
        "decision": "AF when pooled[1] - pooled[0] > threshold (integer margin from tpu_model.forward)",
        "threshold": trans["threshold"],
        "normaliser": norm, "features": data.features,
        "preprocessing": {k: manifest[k] for k in ("lead", "fs", "beats_per_sequence", "hop_beats",
                                                   "lookback_beats", "beat_window_ms", "band_pass_hz",
                                                   "labels", "excluded")},
        "split": {"folds": folds, "final_model": roles},
        "validation": trans["val"], "exact_check": trans["exact_check"],
        "baseline": {k: base[k] for k in ("weights", "bias", "mean", "std", "threshold", "val")},
        "cross_validation": "experiments/ecg_af/summary.json",
        "args": vars(args),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=lambda o: str(o) if isinstance(o, Path) else float(o)) + "\n")
    print(f"validation: {fmt(trans['val'])}; exact {trans['exact_check']}; exported to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "weights" / "ecg_af")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--qat-steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=750)
    parser.add_argument("--val-stride", type=int, default=4, help="evaluate every Nth validation sequence while training")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="all")
    parser.add_argument("--tag", default="", help="write fold results to experiments/ecg_af/<tag>/")
    args = parser.parse_args()
    global OUT
    OUT = OUT / args.tag if args.tag else OUT
    if args.summary:
        summary()
    elif args.final:
        final(args)
    elif args.fold is not None:
        run_fold(args)
    else:
        parser.error("give --fold N, --summary or --final")


if __name__ == "__main__":
    main()
