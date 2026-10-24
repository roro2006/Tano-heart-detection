"""Stream heart-sound recordings through the FPGA front end and model.

    python python/heart_host.py COM4 a0001            # one PhysioNet recording
    python python/heart_host.py COM4 --split test     # score the test split on the board

Each recording is sent as raw 2 kHz samples (AUDIO); the board filters,
windows, runs the transformer on every 2.048 s window and keeps the sum of
(pooled[1] - pooled[0]). Positive means "abnormal: refer". Every result is
checked against heart_features + tpu_model.forward on the PC.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import heart_dataset
import tpu_model
from host import TpuLink, audio_totals

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port")
    parser.add_argument("names", nargs="*", help="PhysioNet recording names, e.g. a0001")
    parser.add_argument("--split", choices=heart_dataset.SPLITS, help="stream every recording of a split")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-check", action="store_true", help="skip the PC reference (faster)")
    args = parser.parse_args()

    model = tpu_model.load(ROOT / "weights" / "heart")
    items = {r.name: r for r in heart_dataset.recordings()}
    if args.split:
        split = heart_dataset.assign_splits(list(items.values()))
        names = [n for n in items if split[n] == args.split]
    else:
        names = args.names
    names = names[:args.limit or None]
    if not names:
        parser.error("give recording names or --split")

    tp = fn = tn = fp = mismatches = 0
    start = time.perf_counter()
    with TpuLink(args.port) as tpu:
        for name in names:
            record = items[name]
            samples = heart_dataset.read_samples(record.path)
            windows, margin, last = tpu.stream(samples)
            abnormal = margin > 0
            if not args.no_check and (windows, margin, last) != audio_totals(model, samples):
                mismatches += 1
                print(f"{name}: FPGA {windows, margin, last} != reference {audio_totals(model, samples)}")
            tp += abnormal and record.label == 1
            fn += not abnormal and record.label == 1
            tn += not abnormal and record.label == 0
            fp += abnormal and record.label == 0
            if len(names) <= 20:
                print(f"{name} ({len(samples) / 2000:.1f} s, label {'abnormal' if record.label else 'normal'}): "
                      f"{windows} windows, margin {margin} -> {'ABNORMAL, refer' if abnormal else 'normal'}")
    elapsed = time.perf_counter() - start
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    print(f"{len(names)} recordings in {elapsed:.0f} s: sensitivity {se:.1%}, specificity {sp:.1%}, "
          f"balanced {(se + sp) / 2:.1%}"
          + ("" if args.no_check else f"; {len(names) - mismatches}/{len(names)} identical to the PC reference"))
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
