"""Run the ECG AF model on the FPGA and check it against the PC.

    python python/ecg_host.py COM4 --records 04015 08455 --limit 200
    python python/ecg_host.py --oracle --records 04015      # no board: PC reference only

The board keeps its baked heart-sound model; this loads weights/ecg_af/ at
runtime (LOAD, about 0.6 s; gone at the next reset or bitstream load) and
sends 16-beat sequences as INFER requests. Beat detection and features run
on the PC (ecg_dataset.py), the transformer on the chip. Every board answer
is compared with tpu_model.forward. Reload the heart bitstream afterwards
(or power-cycle a flashed board) to get the heart model back.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ecg_dataset
import tpu_model

ROOT = Path(__file__).resolve().parents[1]


def quantise(X: np.ndarray, norm: dict) -> np.ndarray:
    z = np.clip((X - np.array(norm["centre"])) / np.array(norm["scale"]), -norm["clip"], norm["clip"])
    q = np.round(z * 256).astype(np.int64)
    q[..., norm.get("zeroed", [])] = 0
    return q


def margin(logits) -> int:
    return sum(row[1] for row in logits) - sum(row[0] for row in logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", nargs="?")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "ecg_af")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0, help="sequences per record, spread evenly (0 = all)")
    args = parser.parse_args()
    if not args.port and not args.oracle:
        parser.error("give a port or --oracle")

    config = json.loads((args.weights / "config.json").read_text())
    model = tpu_model.load(args.weights)
    threshold = config["threshold"]
    held_out = set(config["split"]["final_model"]["val"])
    link = None
    if not args.oracle:
        from host import TpuLink
        link = TpuLink(args.port)
        t = time.perf_counter()
        link.load_model(model)
        print(f"loaded weights/ecg_af into the board in {time.perf_counter() - t:.1f} s")

    tp = fn = tn = fp = mismatches = sent = 0
    chip_cycles = []
    try:
        for record in args.records:
            r = ecg_dataset.process(record)
            idx = np.arange(len(r["y"]))
            if args.limit and len(idx) > args.limit:
                idx = idx[np.linspace(0, len(idx) - 1, args.limit).astype(int)]
            Q = quantise(r["X"][idx], config["normaliser"])
            for q, label in zip(Q, r["y"][idx]):
                features = q.tolist()
                reference = margin(tpu_model.forward(model, features))
                if link:
                    _, cycles, logits = link.infer(features)
                    chip_cycles.append(cycles)
                    got = margin(logits)
                    mismatches += got != reference
                else:
                    got = reference
                af = got > threshold
                tp += af and label == 1; fn += (not af) and label == 1
                tn += (not af) and label == 0; fp += af and label == 0
                sent += 1
            note = "validation patient for the final model" if record in held_out else "training patient for the final model"
            print(f"{record} ({note}): {len(idx)} sequences, AF {int((r['y'][idx] == 1).sum())}")
    finally:
        if link:
            link.close()
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    print(f"{sent} sequences: sensitivity {se:.1%}, specificity {sp:.1%} at threshold {threshold:g}"
          + (f"; {sent - mismatches}/{sent} identical to the PC reference; "
             f"{np.mean(chip_cycles):.0f} cycles = {np.mean(chip_cycles) / 27e3:.2f} ms per inference" if link else ""))
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
