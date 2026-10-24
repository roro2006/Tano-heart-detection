"""Deterministic 16x16 pattern dataset generator.

Each record holds the pattern ``label``, the noisy ``pixels`` (shifted by up
to two cells, 3 % of the lit pixels dropped) and the clean shifted ``target``
that the model learns to reconstruct.
"""
from __future__ import annotations
import random
from pathlib import Path
import json

Grid = list[list[int]]


def patterns() -> dict[str, Grid]:
    out = {}
    for name, fn in {
        "cross": lambda r, c: r == 7 or r == 8 or c == 7 or c == 8,
        "box": lambda r, c: r in (2, 13) or c in (2, 13),
        "x": lambda r, c: r == c or r + c == 15,
        "plus": lambda r, c: r in (7, 8) or c in (7, 8),
    }.items():
        out[name] = [[int(fn(r, c)) for c in range(16)] for r in range(16)]
    return out


def shifted(image: Grid, dr: int, dc: int) -> Grid:
    """Move a pattern down by dr and right by dc, filling with zeros."""
    return [[image[r - dr][c - dc] if 0 <= r - dr < 16 and 0 <= c - dc < 16 else 0
             for c in range(16)] for r in range(16)]


def corrupt(rng: random.Random, clean: Grid, drop: float = 0.03, salt: float = 0.0) -> Grid:
    """Drop lit pixels with probability ``drop`` and light dark ones with ``salt``."""
    return [[int(rng.random() > drop) if value else int(salt > 0 and rng.random() < salt)
             for value in row] for row in clean]


def build(path: Path, copies: int = 300, seed: int = 7) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for name, image in patterns().items():
        for _ in range(copies):
            target = shifted(image, rng.randint(-2, 2), rng.randint(-2, 2))
            records.append({"label": name, "pixels": corrupt(rng, target), "target": target})
    path.write_text(json.dumps(records))


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def split(records: list[dict], every: int = 5) -> tuple[list[dict], list[dict]]:
    """Hold out every ``every``-th record; return (train, test)."""
    return ([r for i, r in enumerate(records) if i % every],
            [r for i, r in enumerate(records) if i % every == 0])


def hard_set(count: int = 200, seed: int = 11, drop: float = 0.15, salt: float = 0.02) -> list[dict]:
    """Held-out records with far more noise than build() uses."""
    rng = random.Random(seed)
    base = patterns()
    records = []
    for i in range(count):
        name = list(base)[i % len(base)]
        target = shifted(base[name], rng.randint(-2, 2), rng.randint(-2, 2))
        records.append({"label": name, "pixels": corrupt(rng, target, drop, salt), "target": target})
    return records


if __name__ == "__main__":
    build(Path("data/patterns.json"))
