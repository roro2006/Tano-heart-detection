from __future__ import annotations
from typing import Sequence
from q8_attention import matmul_q8_8, sat16

def tiled_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    if len(a[0]) != len(b) or len(a) % 8 or len(b[0]) % 4:
        raise ValueError("dimensions must be 8-row and 4-column tile aligned")
    # The PE accumulators are 40 bits wide, so saturation happens once at writeback.
    out = [[0] * len(b[0]) for _ in a]
    for r0 in range(0, len(a), 8):
        for c0 in range(0, len(b[0]), 4):
            for r in range(r0, min(r0 + 8, len(a))):
                for c in range(c0, c0 + 4):
                    out[r][c] = sat16(sum((a[r][k] * b[k][c]) >> 8 for k in range(len(b))))
    return out

def compare(a, b):
    return tiled_matmul(a, b) == matmul_q8_8(a, b)
