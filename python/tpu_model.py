"""Bit-exact spec of the model that runs entirely on the FPGA (hardware v2).

Every value is a Q8.8 int16 and every op is integer, so the RTL can match it
exactly:

    X = sat16(sat16(F Wemb) + POS)                 F: 16 tokens x 16 features
    per layer:
        R = sat16(X + attention_q8_8(X, Wq, Wk, Wv, Wo))
        N = layer_norm_int(R)
        H = relu(N W1)
        X = sat16(H W2 + N)
    logits = X Wout                                16 tokens x 16 outputs
    class  = argmax_j sum_t logits[t][j]           first maximum wins

Matmuls are matmul_q8_8 (products floor-shifted by 8, saturated at the end).
Weights are int8 with a per-matrix left shift: the Q8.8 value is w8 << lsh,
which is what the FPGA feeds into the systolic array. POS is a Q8.8 table
stored on chip (sinusoidal for the pattern denoiser, learnable in general).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from q8_attention import attention_q8_8, matmul_q8_8, sat16

TOKENS, WIDTH, FEATURES, OUTPUTS, LAYERS = 16, 64, 16, 16, 2
LAYER_MATRICES = ("q", "k", "v", "o", "w1", "w2")
SLOT_EMBED, SLOT_OUTPUT = 12, 13
SLOTS = 14
MAX_LSH = 7

Matrix = list[list[int]]

# round(2^24 / sqrt(t*256 + 128)) for t in [256, 1024): 1/sqrt of a value
# normalised to [2^16, 2^18), sampled at the centre of each 256-wide bucket.
RSQRT_LUT = tuple(round((1 << 24) / math.sqrt(t * 256 + 128)) for t in range(256, 1024))


@dataclass(frozen=True)
class Int8Matrix:
    """int8 values whose Q8.8 meaning is value << lsh."""

    values: tuple[tuple[int, ...], ...]
    lsh: int

    def __post_init__(self):
        if not 0 <= self.lsh <= MAX_LSH:
            raise ValueError(f"lsh {self.lsh} out of range")
        if any(not -128 <= v <= 127 for row in self.values for v in row):
            raise ValueError("values must be int8")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.values), len(self.values[0])

    def q8_8(self) -> Matrix:
        return [[v << self.lsh for v in row] for row in self.values]


def quantize(matrix: Sequence[Sequence[float]], scale: float = 256.0) -> Int8Matrix:
    """Quantise real weights (or Q8.8 ints with scale=1) to int8 with the smallest shift that fits."""
    peak = max(abs(v) * scale for row in matrix for v in row)
    lsh = 0
    while lsh < MAX_LSH and peak / (1 << lsh) > 127.5:
        lsh += 1
    values = tuple(tuple(max(-128, min(127, round(v * scale / (1 << lsh)))) for v in row)
                   for row in matrix)
    return Int8Matrix(values, lsh)


@dataclass(frozen=True)
class TpuModel:
    embed: Int8Matrix                       # FEATURES x WIDTH
    layers: tuple[dict[str, Int8Matrix], ...]
    output: Int8Matrix                      # WIDTH x OUTPUTS
    position: tuple[tuple[int, ...], ...]   # TOKENS x WIDTH, Q8.8

    def slot(self, index: int) -> Int8Matrix:
        if index == SLOT_EMBED:
            return self.embed
        if index == SLOT_OUTPUT:
            return self.output
        return self.layers[index // 6][LAYER_MATRICES[index % 6]]


def layer_norm_int(x: Sequence[Sequence[int]]) -> Matrix:
    """Integer layer norm over each 64-wide row: round(c * 2048 / sqrt(sum c^2)).

    c = r - floor(mean). For 64 values the real-valued layer norm in Q8.8 is
    exactly c * 2048 / sqrt(sum c^2). The sum of squares is normalised by an
    even shift into [2^16, 2^18) so a 768-entry table gives 1/sqrt.
    """
    out = []
    for row in x:
        mean = sum(row) >> 6
        c = [v - mean for v in row]
        variance = sum(v * v for v in c)
        k = (variance.bit_length() - 17) >> 1
        normalised = variance >> (2 * k) if k >= 0 else variance << (-2 * k)
        inv = RSQRT_LUT[(normalised >> 8) - 256] if variance else 0
        shift = 13 + k
        out.append([sat16((v * inv + (1 << (shift - 1))) >> shift) for v in c])
    return out


def _add(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> Matrix:
    return [[sat16(x + y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def embed(model: TpuModel, features: Sequence[Sequence[int]]) -> Matrix:
    if len(features) != TOKENS or any(len(row) != FEATURES for row in features):
        raise ValueError("features must be 16 tokens x 16 features")
    return _add(matmul_q8_8(features, model.embed.q8_8()), model.position)


def layer(x: Matrix, weights: dict[str, Int8Matrix]) -> Matrix:
    w = {name: weights[name].q8_8() for name in LAYER_MATRICES}
    residual = _add(x, attention_q8_8(x, w["q"], w["k"], w["v"], w["o"]))
    normalised = layer_norm_int(residual)
    hidden = [[max(0, v) for v in row] for row in matmul_q8_8(normalised, w["w1"])]
    return _add(matmul_q8_8(hidden, w["w2"]), normalised)


def forward(model: TpuModel, features: Sequence[Sequence[int]]) -> Matrix:
    """Return the 16x16 output logits."""
    x = embed(model, features)
    for weights in model.layers:
        x = layer(x, weights)
    return matmul_q8_8(x, model.output.q8_8())


def predicted_class(logits: Sequence[Sequence[int]]) -> int:
    totals = [sum(row[j] for row in logits) for j in range(OUTPUTS)]
    return totals.index(max(totals))


def grid_features(grid: Sequence[Sequence[int]]) -> Matrix:
    """A 16x16 binary grid as features: pixel 1 -> 1.0 in Q8.8."""
    return [[256 if p else 0 for p in row] for row in grid]


def pixels(logits: Sequence[Sequence[int]]) -> Matrix:
    return [[int(v >= 0) for v in row] for row in logits]


def sinusoidal_position() -> tuple[tuple[int, ...], ...]:
    from model_export import positional_embedding
    return tuple(tuple(row) for row in positional_embedding(TOKENS, WIDTH))


def identity_embed() -> Int8Matrix:
    """Wemb that copies feature c into channel c (1.0 = 64 << 2)."""
    return Int8Matrix(tuple(tuple(64 if r == c else 0 for c in range(WIDTH)) for r in range(FEATURES)), 2)


# Export: one manifest plus the files the RTL bakes into BSRAM initial values.

MANIFEST = "tpu_model.json"
W_DEPTH = 12 * 1024 + 2 * 256


def weight_words(model: TpuModel) -> list[int]:
    """The weight memory image: 32-bit words of four int8 lanes (lane 0 low).

    Slot s < 12 starts at s*1024, the embedding at 12*1024 and the output at
    12*1024 + 256. Word base + cg*K + k holds W[k][4cg .. 4cg+3] for a K-row
    matrix, the B-edge column group the systolic array consumes.
    """
    words = [0] * W_DEPTH
    for slot in range(SLOTS):
        matrix = model.slot(slot)
        rows, cols = matrix.shape
        base = slot * 1024 if slot < SLOT_EMBED else 12 * 1024 + (slot - SLOT_EMBED) * 256
        for k in range(rows):
            for cg in range(cols // 4):
                word = 0
                for lane in range(4):
                    word |= (matrix.values[k][4 * cg + lane] & 0xFF) << (8 * lane)
                words[base + cg * rows + k] = word
    return words


def position_words(model: TpuModel) -> list[int]:
    """POS in A-memory layout: word g*64 + k holds POS[8g + l][k] in 16-bit lane l."""
    words = []
    for g in range(2):
        for k in range(WIDTH):
            word = 0
            for lane in range(8):
                word |= (model.position[8 * g + lane][k] & 0xFFFF) << (16 * lane)
            words.append(word)
    return words


def export(directory: Path, model: TpuModel, meta: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tpu_weights.mem").write_text("".join(f"{w:08x}\n" for w in weight_words(model)))
    (directory / "tpu_lsh.mem").write_text("".join(f"{model.slot(s).lsh:x}\n" for s in range(SLOTS)) +
                                           "0\n0\n")
    (directory / "tpu_position.mem").write_text("".join(f"{w:032x}\n" for w in position_words(model)))
    manifest = {
        "embed": _matrix_json(model.embed),
        "layers": [{name: _matrix_json(m) for name, m in weights.items()} for weights in model.layers],
        "output": _matrix_json(model.output),
        "position": [list(row) for row in model.position],
        **(meta or {}),
    }
    (directory / MANIFEST).write_text(json.dumps(manifest) + "\n")


def _matrix_json(matrix: Int8Matrix) -> dict:
    return {"lsh": matrix.lsh, "values": [list(row) for row in matrix.values]}


def _matrix_from_json(data: dict) -> Int8Matrix:
    return Int8Matrix(tuple(tuple(row) for row in data["values"]), data["lsh"])


def load(directory: Path) -> TpuModel:
    data = json.loads((directory / MANIFEST).read_text())
    return TpuModel(
        embed=_matrix_from_json(data["embed"]),
        layers=tuple({name: _matrix_from_json(m) for name, m in weights.items()}
                     for weights in data["layers"]),
        output=_matrix_from_json(data["output"]),
        position=tuple(tuple(row) for row in data["position"]),
    )


def from_quant_model(model) -> TpuModel:
    """Convert a model_export.QuantModel (Q8.8 weights, grid input) to the on-chip model."""
    def layer_weights(index: int) -> dict[str, Int8Matrix]:
        w1, w2 = model.feed_forward[index]
        return {**{name: quantize(model.attention[index][name], 1.0) for name in "qkvo"},
                "w1": quantize(w1, 1.0), "w2": quantize(w2, 1.0)}
    return TpuModel(identity_embed(), tuple(layer_weights(i) for i in range(len(model.attention))),
                    quantize(model.output, 1.0), sinusoidal_position())


def write_rsqrt_rom(path: Path) -> None:
    """1024-deep ROM for the RTL: RSQRT_LUT, then zero padding."""
    path.write_text("".join(f"{v:05x}\n" for v in RSQRT_LUT) + "".join("00000\n" for _ in range(256)))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Convert trained Q8.8 weights to the on-chip model")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--weights", type=Path, default=root / "weights")
    args = parser.parse_args()
    from model_export import load_model
    model = from_quant_model(load_model(args.weights))
    export(args.weights, model, {"source": "model_export.QuantModel via tpu_model.from_quant_model"})
    write_rsqrt_rom(root / "rtl" / "rsqrt_rom.mem")
    print(f"wrote {args.weights / MANIFEST} and the BSRAM images; lsh per slot "
          f"{[model.slot(s).lsh for s in range(SLOTS)]}")


if __name__ == "__main__":
    main()
