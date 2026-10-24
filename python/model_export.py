"""Board-independent Q8.8 model export and PC-side layer completion.

The module accepts plain Python matrices so it remains usable without PyTorch.
Training can be performed externally; this tool defines the exact export
format and the inference-side residual/layer-norm/feed-forward contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from q8_attention import attention_q8_8, q8_8, sat16


def write_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    rows = list(matrix)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be non-empty and rectangular")
    path.write_text("".join(f"{value & 0xffff:04x}\n" for row in rows for value in row))


def read_matrix(path: Path, rows: int, cols: int) -> list[list[int]]:
    values = [int(line, 16) for line in path.read_text().split()]
    if len(values) != rows * cols:
        raise ValueError(f"{path}: expected {rows * cols} values, got {len(values)}")
    signed = [value - 0x10000 if value & 0x8000 else value for value in values]
    return [signed[row * cols:(row + 1) * cols] for row in range(rows)]


def layer_norm_q8_8(x: Sequence[Sequence[int]], eps: float = 1e-5) -> list[list[int]]:
    result = []
    for row in x:
        mean = sum(row) / len(row)
        # Variance of the real values (ints are scaled by 256, so by 65536).
        variance = sum((value - mean) ** 2 for value in row) / len(row) / 65536.0
        scale = 1.0 / (variance + eps) ** 0.5
        result.append([q8_8(((value - mean) / 256.0) * scale) for value in row])
    return result


def relu_q8_8(x: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[max(0, value) for value in row] for row in x]


def finish_layer(
    x: Sequence[Sequence[int]],
    attention: Sequence[Sequence[int]],
    w1: Sequence[Sequence[int]],
    w2: Sequence[Sequence[int]],
) -> list[list[int]]:
    residual = [[sat16(a + b) for a, b in zip(left, right)] for left, right in zip(x, attention)]
    normalized = layer_norm_q8_8(residual)
    hidden = relu_q8_8(_matmul(normalized, w1))
    return [[sat16(a + b) for a, b in zip(row, skip)] for row, skip in
            zip(_matmul(hidden, w2), normalized)]


def _matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    bt = list(zip(*b))
    return [[sat16(sum((left * right) >> 8 for left, right in zip(row, col)))
             for col in bt] for row in a]


def export_layer(directory: Path, layer: int, weights: dict[str, Sequence[Sequence[int]]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("q", "k", "v", "o"):
        if name not in weights:
            raise ValueError(f"missing attention weight {name}")
        write_matrix(directory / f"layer{layer}_{name}.mem", weights[name])


@dataclass(frozen=True)
class QuantModel:
    """Q8.8 weights for full_inference: FPGA attention plus PC-side layers."""

    attention: list[dict[str, list[list[int]]]]
    feed_forward: list[tuple[list[list[int]], list[list[int]]]]
    output: list[list[int]]


MANIFEST = "model.json"


def export_model(directory: Path, model: QuantModel, meta: dict | None = None) -> None:
    """Write every matrix as a .mem file plus a manifest of their shapes.

    The attention files are the ones TpuLink.load_weights sends with LOAD;
    w1/w2 and the output projection stay on the PC.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for layer, weights in enumerate(model.attention):
        export_layer(directory, layer, weights)
        shapes.update({f"layer{layer}_{name}": [64, 64] for name in ("q", "k", "v", "o")})
    for layer, (w1, w2) in enumerate(model.feed_forward):
        for name, matrix in (("w1", w1), ("w2", w2)):
            write_matrix(directory / f"layer{layer}_{name}.mem", matrix)
            shapes[f"layer{layer}_{name}"] = [len(matrix), len(matrix[0])]
    write_matrix(directory / "output.mem", model.output)
    shapes["output"] = [len(model.output), len(model.output[0])]
    manifest = {"layers": len(model.attention), "shapes": shapes, **(meta or {})}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def load_model(directory: Path) -> QuantModel:
    manifest = json.loads((directory / MANIFEST).read_text())
    shapes = manifest["shapes"]

    def matrix(name: str) -> list[list[int]]:
        return read_matrix(directory / f"{name}.mem", *shapes[name])

    layers = range(manifest["layers"])
    return QuantModel(
        attention=[{name: matrix(f"layer{layer}_{name}") for name in ("q", "k", "v", "o")}
                   for layer in layers],
        feed_forward=[(matrix(f"layer{layer}_w1"), matrix(f"layer{layer}_w2")) for layer in layers],
        output=matrix("output"),
    )


def save_vector(path: Path, x: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]) -> None:
    path.write_text(json.dumps({"input": x, "expected": expected}, separators=(",", ":")))


def positional_embedding(rows: int = 16, width: int = 64) -> list[list[int]]:
    """Return deterministic sinusoidal positional encodings in Q8.8."""
    import math
    return [[q8_8(math.sin(row / (10000 ** (2 * (col // 2) / width)))
                  if col % 2 == 0 else
                  math.cos(row / (10000 ** (2 * (col // 2) / width))))
             for col in range(width)] for row in range(rows)]


def embed_grid(grid: Sequence[Sequence[int]], width: int = 64) -> list[list[int]]:
    if len(grid) != 16 or any(len(row) != 16 for row in grid):
        raise ValueError("grid must be 16x16")
    position = positional_embedding(16, width)
    return [[sat16((q8_8(grid[row][col]) if col < 16 else 0) + position[row][col])
             for col in range(width)] for row in range(16)]


def sigmoid_q8_8(value: int) -> int:
    import math
    return q8_8(1.0 / (1.0 + math.exp(-value / 256.0)))


def output_projection(x: Sequence[Sequence[int]], weight: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[sigmoid_q8_8(value) for value in row] for row in _matmul(x, weight)]


def full_inference(
    grid: Sequence[Sequence[int]],
    attention_weights: Sequence[dict[str, Sequence[Sequence[int]]]],
    feed_forward: Sequence[tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]],
    output_weight: Sequence[Sequence[int]],
    attention_transport=None,
) -> list[list[int]]:
    """Run embedding, two attention round trips, transformer completion, and output.

    attention_transport(layer, x) runs one layer on the FPGA, e.g.
    ``TpuLink(port).attention``; it defaults to the bit-exact software oracle.
    """
    x = embed_grid(grid)
    for layer, weights in enumerate(attention_weights):
        if attention_transport is None:
            attended = attention_q8_8(x, weights["q"], weights["k"], weights["v"], weights["o"])
        else:
            attended = attention_transport(layer, x)
        x = finish_layer(x, attended, *feed_forward[layer])
    return output_projection(x, output_weight)


def encode_q8_matrix(matrix: Sequence[Sequence[int]]) -> bytes:
    import struct
    return b"".join(struct.pack("<h", value) for row in matrix for value in row)
