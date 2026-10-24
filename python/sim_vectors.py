"""Build the UART byte script for sim/tb_tpu_top.sv from the Python oracle.

Each output line is three hex digits: 0xx drives byte xx into the FPGA, 1xx
expects byte xx back, 2xx expects any byte (cycle counts and the checksums
that cover them), and fff ends the script.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dataset
import tpu_model
from host import (OP_ATTN, OP_ECHO, OP_ERROR, OP_INFER, OP_INFO, OP_LOAD, OP_LOADPOS,
                  attention_request, encode_q8_matrix, infer_request, infer_response,
                  load_request, position_request, random_matrix, random_model, request,
                  response)
from q8_attention import attention_q8_8, dot_q8_8, matmul_q8_8, sat16

ROOT = Path(__file__).resolve().parents[1]
ANY = None
Expected = list  # of byte values or ANY


def peaks_at_last_key(x, weights) -> bool:
    """True when some row in each 8-row group has its first score maximum at key 15."""
    q, k = matmul_q8_8(x, weights["q"]), matmul_q8_8(x, weights["k"])
    scores = [[sat16(dot_q8_8(qr, kr) >> 3) for kr in k] for qr in q]
    last = [row.index(max(row)) == 15 for row in scores]
    return any(last[:8]) and any(last[8:])


def exact(frame: bytes) -> Expected:
    return list(frame)


def with_cycles(frame: bytes, offset: int) -> Expected:
    """Mask the 4 cycle-count bytes at payload offset `offset` and the checksum."""
    expected = list(frame)
    for i in range(4 + offset, 4 + offset + 4):
        expected[i] = ANY
    expected[-1] = ANY
    return expected


def build(seed: int = 5) -> list[tuple[bytes, Expected]]:
    rng = random.Random(seed)
    steps = [(request(OP_INFO), exact(response(OP_INFO, 0, b"TPU\x02" + bytes(4)))),
             (request(OP_ECHO, 7, b"hello"), exact(response(OP_ECHO, 0, b"hello")))]
    corrupt = bytearray(request(OP_ECHO, 0, b"xyz"))
    corrupt[-1] ^= 0xFF
    steps.append((bytes(corrupt), exact(response(OP_ERROR, 1))))

    # The model baked into the BSRAM initial values, before any LOAD.
    baked = tpu_model.load(ROOT / "weights")
    for record in dataset.hard_set(2, seed=seed):
        features = tpu_model.grid_features(record["pixels"])
        steps.append((infer_request(features),
                      with_cycles(infer_response(tpu_model.forward(baked, features), 0), 3)))

    # A random model loaded at run time, every slot and the position table.
    model = random_model(rng)
    for slot in range(tpu_model.SLOTS):
        steps.append((load_request(slot, model.slot(slot)), exact(response(OP_LOAD))))
    steps.append((position_request(model.position), exact(response(OP_LOADPOS))))
    for span in (512, 20000):
        features = random_matrix(rng, 16, 16, span)
        steps.append((infer_request(features),
                      with_cycles(infer_response(tpu_model.forward(model, features), 0), 3)))

    # Attention alone, on the loaded int8 weights. The input peaks at the last
    # key in both row groups (the softmax write/fix-up boundary).
    layer = 1
    weights = {name: model.layers[layer][name].q8_8() for name in "qkvo"}
    x = random_matrix(rng, 16, 64, 512)
    while not peaks_at_last_key(x, weights):
        x = random_matrix(rng, 16, 64, 512)
    expected = attention_q8_8(x, weights["q"], weights["k"], weights["v"], weights["o"])
    steps.append((attention_request(layer, x), exact(response(OP_ATTN, 0, encode_q8_matrix(expected)))))
    steps.append((request(OP_ATTN, layer, b"\x00" * 10), exact(response(OP_ATTN, 1))))
    steps.append((request(OP_INFER, 0, b"\x00" * 10), exact(response(OP_INFER, 1))))
    steps.append((request(OP_INFO), with_cycles(response(OP_INFO, 0, b"TPU\x02" + bytes(4)), 4)))
    return steps


def a_words(matrix, lane_bits: int = 18) -> list[int]:
    """A 16x64 matrix in A-memory layout: word g*64 + k holds M[8g + l][k] in lane l."""
    words = []
    for g in range(2):
        for k in range(len(matrix[0])):
            word = 0
            for lane in range(8):
                word |= (matrix[8 * g + lane][k] & ((1 << lane_bits) - 1)) << (lane_bits * lane)
            words.append(word)
    return words


def build_ln(seed: int = 3) -> None:
    """Vectors for sim/tb_ln_unit.sv: R at A address 128, expected N at 0."""
    from tpu_model import layer_norm_int
    rng = random.Random(seed)
    rows = []
    for scale in (3, 40, 700, 9000, 32767):
        rows += [[rng.randint(-scale, scale) for _ in range(64)] for _ in range(4)]
    rows[0] = [123] * 64                      # zero variance
    rows[1] = [-7] * 63 + [-6]                # tiny variance
    rows[2] = [32767, -32768] * 32            # full scale
    image = [0] * 512
    image[128:256] = a_words(rows[:16])
    expected = a_words(layer_norm_int(rows[:16]))
    (ROOT / "sim" / "ln_input.mem").write_text("".join(f"{w:036x}\n" for w in image))
    (ROOT / "sim" / "ln_expected.mem").write_text("".join(f"{w:036x}\n" for w in expected))
    print("wrote sim/ln_input.mem and sim/ln_expected.mem")


def main() -> None:
    if "--ln" in sys.argv:
        build_ln()
        return
    lines = []
    for sent, expected in build():
        lines += [f"0{b:02x}" for b in sent]
        lines += ["200" if b is ANY else f"1{b:02x}" for b in expected]
    lines.append("fff")
    out = ROOT / "sim" / "tpu_top_vectors.mem"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} entries)")


if __name__ == "__main__":
    main()
