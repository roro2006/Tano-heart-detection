"""Bit-exact Q8.8 attention reference for the Tang Nano 20K design."""

from __future__ import annotations

import math
import struct
from typing import Iterable, Sequence

Q = 8
SCALE = 1 << Q
I16_MIN, I16_MAX = -(1 << 15), (1 << 15) - 1


def sat16(value: int) -> int:
    return max(I16_MIN, min(I16_MAX, int(value)))


def q8_8(value: float) -> int:
    """Round a real value to signed Q8.8 with saturation."""
    return sat16(round(value * SCALE))


def dot_q8_8(a: Sequence[int], b: Sequence[int]) -> int:
    """Return a dot product in Q8.8 (products are shifted before summing)."""
    if len(a) != len(b):
        raise ValueError("dot-product operands must have equal length")
    return sum((int(x) * int(y)) >> Q for x, y in zip(a, b))


def matmul_q8_8(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    """Saturating matrix multiplication with Q8.8 operands and result."""
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("incompatible matrix dimensions")
    bt = list(zip(*b))
    return [[sat16(dot_q8_8(row, col)) for col in bt] for row in a]


def _exp_lut() -> tuple[int, ...]:
    # Q0.16 values for exp(-n/16), n in [0, 255]. This is also the format
    # recommended for a ROM generated into softmax_exp.mem.
    return tuple(round(math.exp(-n / 16.0) * (1 << 16)) for n in range(256))


EXP_LUT = _exp_lut()


def softmax_q8_8(scores: Sequence[Sequence[int]]) -> list[list[int]]:
    """Integer softmax, returning Q0.16 probabilities (rows sum to 65536)."""
    result: list[list[int]] = []
    for row in scores:
        if not row:
            raise ValueError("softmax rows cannot be empty")
        maximum = max(row)
        exps = [EXP_LUT[min(255, max(0, (maximum - value) >> 4))] for value in row]
        total = sum(exps)
        if total == 0:
            raise ArithmeticError("softmax reciprocal denominator is zero")
        probs = [(value << 16) // total for value in exps]
        # Assign truncation remainder to the maximum, preserving row sum.
        probs[row.index(maximum)] += (1 << 16) - sum(probs)
        result.append(probs)
    return result


def attention_q8_8(
    x: Sequence[Sequence[int]],
    wq: Sequence[Sequence[int]],
    wk: Sequence[Sequence[int]],
    wv: Sequence[Sequence[int]],
    wo: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """Compute one attention layer using the same shifts as the RTL plan."""
    q, k, v = matmul_q8_8(x, wq), matmul_q8_8(x, wk), matmul_q8_8(x, wv)
    # d_k=64, so scaled dot product uses approximately /8 (>> 3).
    scores = [[sat16(dot_q8_8(qr, kr) >> 3) for kr in k] for qr in q]
    probabilities = softmax_q8_8(scores)
    # Q0.16 probability times Q8.8 value returns Q8.8 after >> 16.
    attended = [
        [sat16(sum(p * value for p, value in zip(probability, column)) >> 16)
         for column in zip(*v)]
        for probability in probabilities
    ]
    return matmul_q8_8(attended, wo) if wo is not None else attended


def pack_frame(payload: bytes) -> bytes:
    """Packet format: A5 5A, little-endian payload length, payload, XOR checksum."""
    if len(payload) > 0xFFFF:
        raise ValueError("payload is too large")
    header = b"\xA5\x5A" + struct.pack("<H", len(payload))
    return header + payload + bytes([__import__("functools").reduce(lambda a, b: a ^ b, payload, 0)])


def unpack_frame(frame: bytes) -> bytes:
    if len(frame) < 5 or frame[:2] != b"\xA5\x5A":
        raise ValueError("invalid frame header")
    size = struct.unpack_from("<H", frame, 2)[0]
    if len(frame) != size + 5:
        raise ValueError("invalid frame length")
    payload = frame[4:-1]
    checksum = 0
    for byte in payload:
        checksum ^= byte
    if checksum != frame[-1]:
        raise ValueError("invalid frame checksum")
    return payload

