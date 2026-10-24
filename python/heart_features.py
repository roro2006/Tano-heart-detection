"""Bit-exact integer audio front end for heart sounds (the FPGA will match it).

Input is 16-bit PCM at 2 kHz (the PhysioNet 2016 rate). A window is 4096
samples (2.048 s) = 16 frames of 256 samples, and it becomes the model's
16 tokens x 16 features:

    16 bandpass biquads, centres 25..800 Hz log-spaced, Q = 3, coefficients
    in Q2.16 (18-bit). Direct form I with the bandpass zeros b1 = 0, b2 = -b0,
    keeping 2 fractional bits in y (y is 4x the filtered signal):
        y[n] = sat18((4 g (x[n] - x[n-2]) - a1 y[n-1] - a2 y[n-2] + 2^15) >> 16)
    per frame and band: A = sum |y| over 256 samples
    feature L = log2_q8_8(A): 256 * log2(A) from the MSB position and a
                256-entry fraction table (0 for A = 0)
    per window: F[t][b] = L[t][b] - (sum of the window's 256 L values >> 8)

The mean subtraction removes the recording gain (a gain of 2 adds 256 to
every L). Filters run continuously over a recording, so a window's features
depend on the samples before it.
"""
from __future__ import annotations

import math
from typing import Sequence

RATE = 2000
BANDS = 16
FRAME = 256
FRAMES = 16
WINDOW = FRAME * FRAMES
COEFF_BITS = 16
Y_BITS = 18
Y_FRACTION = 2
Y_MAX, Y_MIN = (1 << (Y_BITS - 1)) - 1, -(1 << (Y_BITS - 1))
LOW_HZ, HIGH_HZ, Q = 25.0, 800.0, 3.0


def band_centres() -> list[float]:
    return [LOW_HZ * (HIGH_HZ / LOW_HZ) ** (k / (BANDS - 1)) for k in range(BANDS)]


def _coefficients() -> tuple[tuple[int, int, int], ...]:
    """(g, a1, a2) in Q2.16 for each band: RBJ constant-0-dB-peak bandpass."""
    scale = 1 << COEFF_BITS
    out = []
    for f0 in band_centres():
        w0 = 2 * math.pi * f0 / RATE
        alpha = math.sin(w0) / (2 * Q)
        a0 = 1 + alpha
        out.append((round(alpha / a0 * scale), round(-2 * math.cos(w0) / a0 * scale),
                    round((1 - alpha) / a0 * scale)))
    return tuple(out)


COEFFS = _coefficients()
# round(256 * log2(1 + f/256)) for f in [0, 256): the fractional part of log2.
LOG2_FRAC = tuple(round(256 * math.log2(1 + f / 256)) for f in range(256))


def sat18(value: int) -> int:
    return max(Y_MIN, min(Y_MAX, value))


def log2_q8_8(value: int) -> int:
    """256 * log2(value) for value >= 1, from the MSB and the next 8 bits; 0 for 0."""
    if value <= 0:
        return 0
    msb = value.bit_length() - 1
    fraction = (value << 8 >> msb) & 0xFF if msb < 8 else (value >> (msb - 8)) & 0xFF
    return (msb << 8) + LOG2_FRAC[fraction]


class FilterBank:
    """Streaming state of the 16 biquads; feed samples, read frame energies."""

    def __init__(self):
        self.x1 = self.x2 = 0
        self.y1 = [0] * BANDS
        self.y2 = [0] * BANDS

    def step(self, x: int) -> list[int]:
        # Two fractional bits: quiet bands keep resolution, and a full-scale
        # 16-bit input at unity gain still fits the 18-bit state.
        diff = (x - self.x2) << Y_FRACTION
        out = []
        for b, (g, a1, a2) in enumerate(COEFFS):
            acc = g * diff - a1 * self.y1[b] - a2 * self.y2[b]
            y = sat18((acc + (1 << (COEFF_BITS - 1))) >> COEFF_BITS)
            self.y2[b], self.y1[b] = self.y1[b], y
            out.append(y)
        self.x2, self.x1 = self.x1, x
        return out


def frame_energies(samples: Sequence[int]) -> list[list[int]]:
    """Sum |y| per band for every complete 256-sample frame."""
    bank = FilterBank()
    frames, acc = [], [0] * BANDS
    for n, x in enumerate(samples):
        for b, y in enumerate(bank.step(x)):
            acc[b] += abs(y)
        if n % FRAME == FRAME - 1:
            frames.append(acc)
            acc = [0] * BANDS
    return frames


def window_features(log_frames: Sequence[Sequence[int]]) -> list[list[int]]:
    """16 frames of log energies -> mean-removed 16x16 features."""
    if len(log_frames) != FRAMES or any(len(row) != BANDS for row in log_frames):
        raise ValueError("a window is 16 frames x 16 bands")
    mean = sum(sum(row) for row in log_frames) >> 8
    return [[v - mean for v in row] for row in log_frames]


def recording_windows(samples: Sequence[int], hop_frames: int = 8,
                      skip_frames: int = 2) -> list[list[list[int]]]:
    """All windows of a recording, starting after `skip_frames` of filter warm-up."""
    logs = [[log2_q8_8(a) for a in row] for row in frame_energies(samples)]
    return [window_features(logs[start:start + FRAMES])
            for start in range(skip_frames, len(logs) - FRAMES + 1, hop_frames)]
