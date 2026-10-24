"""Measure fixed-point LUT softmax error against floating point."""
from __future__ import annotations
import math
import random
from q8_attention import softmax_q8_8

def measure(samples: int = 1000, seed: int = 11) -> float:
    rng = random.Random(seed)
    error = 0.0
    for _ in range(samples):
        row = [rng.randint(-2048, 2048) for _ in range(16)]
        fixed = softmax_q8_8([row])[0]
        maximum = max(row)
        exp_values = [math.exp((value - maximum) / 256.0) for value in row]
        total = sum(exp_values)
        reference = [value / total for value in exp_values]
        error += sum(abs((value / 65536.0) - expected)
                     for value, expected in zip(fixed, reference)) / len(row)
    return error / samples

if __name__ == "__main__":
    print(f"mean absolute probability error: {measure():.6f}")
