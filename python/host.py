"""PC-side transport for the Tang Nano TPU packet protocol (hardware v2).

Request payload is ``op, arg, data``; the response payload is ``op, status,
data``. See rtl/tpu_top.sv for the authoritative command table and
python/tpu_model.py for the model the FPGA runs.
"""

from __future__ import annotations

import argparse
import random
import struct
import time
from pathlib import Path
from typing import Sequence

import tpu_model
from q8_attention import attention_q8_8, pack_frame, unpack_frame
from tpu_model import Int8Matrix, TpuModel

OP_ECHO, OP_LOAD, OP_ATTN, OP_INFO, OP_INFER, OP_LOADPOS, OP_ERROR = (
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xEE)
MATRIX_NAMES = ("q", "k", "v", "o")
TOKENS, WIDTH, FEATURES, OUTPUTS = 16, 64, 16, 16
# Must match BAUD in rtl/tang_nano_20k_top.sv.
DEFAULT_BAUD = 1_000_000
ROOT = Path(__file__).resolve().parents[1]


def read_matrix(path: Path, rows: int, cols: int) -> list[list[int]]:
    values = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    values = [value - 0x10000 if value & 0x8000 else value for value in values]
    if len(values) != rows * cols:
        raise ValueError(f"{path} contains {len(values)} values; expected {rows * cols}")
    return [values[row * cols:(row + 1) * cols] for row in range(rows)]


def encode_q8_matrix(matrix: Sequence[Sequence[int]]) -> bytes:
    return b"".join(struct.pack("<h", value) for row in matrix for value in row)


def decode_q8_matrix(payload: bytes, rows: int, cols: int) -> list[list[int]]:
    if len(payload) != rows * cols * 2:
        raise ValueError("response does not contain the expected matrix")
    values = struct.unpack("<" + "h" * (rows * cols), payload)
    return [list(values[row * cols:(row + 1) * cols]) for row in range(rows)]


def _check_shape(matrix: Sequence[Sequence[int]], rows: int, cols: int, what: str) -> None:
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"{what} must be {rows}x{cols}")


def request(op: int, arg: int = 0, data: bytes = b"") -> bytes:
    return pack_frame(bytes([op, arg]) + data)


def slot_shape(slot: int) -> tuple[int, int]:
    if slot == tpu_model.SLOT_EMBED:
        return FEATURES, WIDTH
    if slot == tpu_model.SLOT_OUTPUT:
        return WIDTH, OUTPUTS
    return WIDTH, WIDTH


def load_request(slot: int, matrix: Int8Matrix) -> bytes:
    if not 0 <= slot < tpu_model.SLOTS:
        raise ValueError(f"slot {slot} out of range")
    _check_shape(matrix.values, *slot_shape(slot), f"slot {slot}")
    data = bytes([matrix.lsh]) + bytes(v & 0xFF for row in matrix.values for v in row)
    return request(OP_LOAD, slot, data)


def attention_request(layer: int, x: Sequence[Sequence[int]]) -> bytes:
    _check_shape(x, TOKENS, WIDTH, "attention input")
    return request(OP_ATTN, layer, encode_q8_matrix(x))


def infer_request(features: Sequence[Sequence[int]]) -> bytes:
    _check_shape(features, TOKENS, FEATURES, "features")
    return request(OP_INFER, 0, encode_q8_matrix(features))


def position_request(position: Sequence[Sequence[int]]) -> bytes:
    _check_shape(position, TOKENS, WIDTH, "position table")
    return request(OP_LOADPOS, 0, encode_q8_matrix(position))


def response(op: int, status: int = 0, data: bytes = b"") -> bytes:
    return pack_frame(bytes([op, status]) + data)


def infer_response(logits: Sequence[Sequence[int]], cycles: int) -> bytes:
    data = bytes([tpu_model.predicted_class(logits)]) + struct.pack("<I", cycles)
    return response(OP_INFER, 0, data + encode_q8_matrix(logits))


class TpuLink:
    """Synchronous request/response client for the FPGA."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = 5.0):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("install pyserial to use the hardware host: python -m pip install pyserial") from exc
        self.link = serial.Serial(port, baudrate=baud, timeout=timeout)
        # The USB bridge can still hold bytes from whatever ran before; let
        # them arrive, then discard them so the first header is ours.
        time.sleep(0.2)
        self.link.reset_input_buffer()

    def close(self) -> None:
        self.link.close()

    def __enter__(self) -> "TpuLink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def transact(self, frame: bytes) -> tuple[int, int, bytes]:
        self.link.write(frame)
        self.link.flush()
        header = self.link.read(4)
        if len(header) != 4 or header[:2] != b"\xA5\x5A":
            raise TimeoutError(f"bad or missing response header: {header.hex()}")
        length = struct.unpack("<H", header[2:])[0]
        body = self.link.read(length + 1)
        if len(body) != length + 1:
            raise TimeoutError("timed out waiting for FPGA response payload")
        payload = unpack_frame(header + body)
        if len(payload) < 2:
            raise ValueError("response payload is too short")
        return payload[0], payload[1], payload[2:]

    def _checked(self, frame: bytes, op: int) -> bytes:
        got_op, status, data = self.transact(frame)
        if got_op == OP_ERROR:
            raise IOError(f"FPGA reported a framing error (status {status})")
        if got_op != op or status != 0:
            raise IOError(f"command {op:#04x} failed: op={got_op:#04x} status={status}")
        return data

    def info(self) -> tuple[bytes, int, int]:
        """Return (magic, hardware version, cycles of the last ATTN/INFER)."""
        data = self._checked(request(OP_INFO), OP_INFO)
        return data[:3], data[3], struct.unpack("<I", data[4:8])[0]

    def echo(self, data: bytes) -> bytes:
        return self._checked(request(OP_ECHO, 0, data), OP_ECHO)

    def load_slot(self, slot: int, matrix: Int8Matrix) -> None:
        self._checked(load_request(slot, matrix), OP_LOAD)

    def load_position(self, position: Sequence[Sequence[int]]) -> None:
        self._checked(position_request(position), OP_LOADPOS)

    def load_model(self, model: TpuModel) -> None:
        """Replace the baked model until the next reset."""
        for slot in range(tpu_model.SLOTS):
            self.load_slot(slot, model.slot(slot))
        self.load_position(model.position)

    def attention(self, layer: int, x: Sequence[Sequence[int]]) -> list[list[int]]:
        data = self._checked(attention_request(layer, x), OP_ATTN)
        return decode_q8_matrix(data, TOKENS, WIDTH)

    def infer(self, features: Sequence[Sequence[int]]) -> tuple[int, int, list[list[int]]]:
        """Run the whole model on the FPGA: (class, compute cycles, 16x16 logits)."""
        data = self._checked(infer_request(features), OP_INFER)
        cycles = struct.unpack("<I", data[1:5])[0]
        return data[0], cycles, decode_q8_matrix(data[5:], TOKENS, OUTPUTS)


def run_once(port: str, baud: int, payload: bytes, timeout: float) -> bytes:
    """Send one raw payload (op, arg, data) and return the raw response payload."""
    with TpuLink(port, baud, timeout) as tpu:
        op, status, data = tpu.transact(pack_frame(payload))
        return bytes([op, status]) + data


def random_matrix(rng: random.Random, rows: int, cols: int, span: int) -> list[list[int]]:
    return [[rng.randint(-span, span) for _ in range(cols)] for _ in range(rows)]


def random_int8(rng: random.Random, rows: int, cols: int, span: int = 40,
                lsh: int | None = None) -> Int8Matrix:
    return Int8Matrix(tuple(tuple(rng.randint(-span, span) for _ in range(cols)) for _ in range(rows)),
                      rng.randint(0, 2) if lsh is None else lsh)


def random_model(rng: random.Random) -> TpuModel:
    layers = tuple({name: random_int8(rng, WIDTH, WIDTH) for name in tpu_model.LAYER_MATRICES}
                   for _ in range(tpu_model.LAYERS))
    return TpuModel(random_int8(rng, FEATURES, WIDTH, 100), layers, random_int8(rng, WIDTH, OUTPUTS),
                    tuple(tuple(row) for row in random_matrix(rng, TOKENS, WIDTH, 300)))


def selftest(tpu: TpuLink, seed: int, runs: int) -> bool:
    rng = random.Random(seed)
    magic, version, _ = tpu.info()
    print(f"info: {magic.decode(errors='replace')} v{version}")
    probe = bytes(rng.randrange(256) for _ in range(256))
    if tpu.echo(probe) != probe:
        print("echo: MISMATCH")
        return False
    print("echo: ok (256 bytes)")
    ok = True

    baked_dir = ROOT / "weights"
    if (baked_dir / tpu_model.MANIFEST).exists():
        # Must run before any LOAD: checks the model built into the bitstream.
        import dataset
        baked = tpu_model.load(baked_dir)
        same = 0
        for record in dataset.hard_set(8, seed=seed):
            features = tpu_model.grid_features(record["pixels"])
            got_class, cycles, logits = tpu.infer(features)
            expected = tpu_model.forward(baked, features)
            same += logits == expected and got_class == tpu_model.predicted_class(expected)
        print(f"baked model: {same}/8 inferences identical to tpu_model.forward "
              f"({cycles} cycles = {cycles / 27e3:.2f} ms at 27 MHz)")
        ok = ok and same == 8

    model = random_model(rng)
    start = time.perf_counter()
    tpu.load_model(model)
    print(f"random model: loaded in {time.perf_counter() - start:.1f} s")
    for run in range(runs):
        features = random_matrix(rng, TOKENS, FEATURES, 512)
        expected = tpu_model.forward(model, features)
        start = time.perf_counter()
        got_class, cycles, logits = tpu.infer(features)
        elapsed = time.perf_counter() - start
        bad = sum(g != e for gr, er in zip(logits, expected) for g, e in zip(gr, er))
        bad += got_class != tpu_model.predicted_class(expected)
        print(f"infer run {run}: {'ok' if bad == 0 else f'{bad} mismatches'} "
              f"({cycles} cycles, {elapsed * 1000:.0f} ms round trip)")
        ok = ok and bad == 0
    for layer in range(tpu_model.LAYERS):
        w = {name: model.layers[layer][name].q8_8() for name in MATRIX_NAMES}
        x = random_matrix(rng, TOKENS, WIDTH, 512)
        bad = sum(g != e for gr, er in zip(tpu.attention(layer, x),
                                           attention_q8_8(x, w["q"], w["k"], w["v"], w["o"]))
                  for g, e in zip(gr, er))
        print(f"attention layer {layer}: {'ok' if bad == 0 else f'{bad} mismatches'}")
        ok = ok and bad == 0
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="UART port, e.g. COM4 or /dev/ttyUSB1")
    parser.add_argument("command", choices=("info", "selftest", "raw"))
    parser.add_argument("payload", type=Path, nargs="?", help="raw payload file for 'raw'")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()
    with TpuLink(args.port, args.baud, args.timeout) as tpu:
        if args.command == "info":
            magic, version, cycles = tpu.info()
            print(f"{magic.decode(errors='replace')} v{version}, last compute {cycles} cycles")
        elif args.command == "selftest":
            raise SystemExit(0 if selftest(tpu, args.seed, args.runs) else 1)
        else:
            if args.payload is None:
                parser.error("raw needs a payload file")
            op, status, data = tpu.transact(pack_frame(args.payload.read_bytes()))
            print(f"op={op:#04x} status={status} {len(data)} data bytes")


if __name__ == "__main__":
    main()
