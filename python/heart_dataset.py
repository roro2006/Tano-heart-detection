"""PhysioNet/CinC 2016 heart sounds -> cached 16x16 feature windows.

    python python/heart_dataset.py            # writes data/heart_windows.pt

Recordings come from data/physionet2016/training-{a..f}/ (REFERENCE.csv:
name, label with -1 normal and 1 abnormal). Every recording becomes windows
of heart_features.recording_windows (2.048 s, hop 1.024 s), labelled
0 = normal, 1 = abnormal.

The split is by recording, stratified by source database and label, so no
recording contributes to both training and test. The six databases were
collected with different equipment and have very different abnormal rates,
so results are also reported per database (a model can learn the source
instead of the heart). Some databases may hold several recordings of one
subject; the public files do not say, so subject-level leakage is possible.
"""
from __future__ import annotations

import array
import csv
import random
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import heart_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "physionet2016"
CACHE = ROOT / "data" / "heart_windows.pt"
SPLITS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class Recording:
    name: str
    database: str
    label: int          # 0 normal, 1 abnormal
    path: Path


def recordings(data: Path = DATA) -> list[Recording]:
    out = []
    for folder in sorted(data.glob("training-*")):
        with open(folder / "REFERENCE.csv", newline="") as handle:
            for name, label in csv.reader(handle):
                if label not in ("-1", "1"):
                    raise ValueError(f"{folder.name}/{name}: unexpected label {label!r}")
                out.append(Recording(name, folder.name[-1], int(label == "1"), folder / f"{name}.wav"))
    return out


def read_samples(path: Path) -> list[int]:
    with wave.open(str(path)) as handle:
        if (handle.getframerate(), handle.getsampwidth(), handle.getnchannels()) != (heart_features.RATE, 2, 1):
            raise ValueError(f"{path}: expected 2 kHz 16-bit mono")
        return list(array.array("h", handle.readframes(handle.getnframes())))


def assign_splits(items: list[Recording], seed: int = 2016) -> dict[str, str]:
    """Recording name -> split, stratified by (database, label)."""
    groups: dict[tuple[str, int], list[str]] = {}
    for r in items:
        groups.setdefault((r.database, r.label), []).append(r.name)
    rng = random.Random(seed)
    out = {}
    for key in sorted(groups):
        names = sorted(groups[key])
        rng.shuffle(names)
        cut1 = round(len(names) * FRACTIONS[0])
        cut2 = cut1 + round(len(names) * FRACTIONS[1])
        for i, name in enumerate(names):
            out[name] = SPLITS[0] if i < cut1 else SPLITS[1] if i < cut2 else SPLITS[2]
    return out


def _windows(path: Path) -> list[list[list[int]]]:
    return heart_features.recording_windows(read_samples(path))


def build(cache: Path = CACHE, workers: int = 8) -> None:
    import torch
    items = recordings()
    split = assign_splits(items)
    with ProcessPoolExecutor(workers) as pool:
        per_recording = list(pool.map(_windows, [r.path for r in items], chunksize=16))
    features, owner = [], []
    for index, windows in enumerate(per_recording):
        features += windows
        owner += [index] * len(windows)
    torch.save({
        "features": torch.tensor(features, dtype=torch.int16),
        "recording": torch.tensor(owner, dtype=torch.int32),
        "names": [r.name for r in items],
        "databases": [r.database for r in items],
        "labels": torch.tensor([r.label for r in items], dtype=torch.int64),
        "splits": [split[r.name] for r in items],
        "spec": {"rate": heart_features.RATE, "window": heart_features.WINDOW, "hop_frames": 8},
    }, cache)
    counts = {s: sum(split[r.name] == s for r in items) for s in SPLITS}
    print(f"{len(items)} recordings {counts}, {len(features)} windows -> {cache}")


if __name__ == "__main__":
    build()
