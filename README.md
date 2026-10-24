# Tano heart detection

A low-cost, offline heart-sound screening device: a microphone on a
stethoscope feeds an audio front end and a small transformer running entirely
on a Sipeed Tang Nano 20K, which flags recordings that may contain a murmur so
health workers know whom to refer. It is a screening prototype, not a
diagnostic device.

This repository starts from
[Tang-Nano-20k-FPGA-TPU](https://github.com/roro2006/Tang-Nano-20k-FPGA-TPU)
at commit `0cf5a87`, the on-chip transformer described below. The heart-sound
work (audio front end, training on public heart-sound recordings, microphone
input and standalone output) is built on top of it here.

## The on-chip transformer

A two-layer, 16-token, 64-wide transformer that runs entirely on the Sipeed
Tang Nano 20K (Gowin GW2AR-LV18QN88C8/I7). The trained weights are baked into
the bitstream, so the board runs the model from power-up: one `INFER` request
sends 16 tokens × 16 features and gets back 16×16 logits and a class, all
computed on the FPGA and bit-exact with `python/tpu_model.py`:

```
X = sat16(F Wemb) + POS                  embedding (Q8.8, int8 weights << shift)
per layer:
  R = X + attention(X)                   Q/K/V, S = (Q K^T) >> 3, softmax, P V, Wo
  N = layer_norm_int(R)                  integer mean/variance, 1/sqrt from a ROM
  X = relu(N W1) W2 + N
logits = X Wout, class = argmax_j sum_t logits[t][j]
```

The bundled model is a pattern denoiser (draw in `python/app.py`): 100%
pixel accuracy on held-out patterns and 99.86% on a harder noise set, with
every FPGA output identical to the Python reference.

## Hardware

* **8×4 systolic array** of 18×18 MAC cells (32 of the 48 `MULT18X18` DSPs).
  Operands are 18 bits so Q0.16 probabilities up to 65536 share the datapath.
  A tile of K terms runs in K + 10 cycles; matrices are tiled 8 rows × 4 columns.
* **On-chip memories** (38 of 46 BSRAMs): 50 KiB of int8 weights (12 layer
  matrices, embedding and output head) with a per-matrix shift, a position
  table, a 512×144-bit activation memory in "8 rows per word" layout feeding
  the array's A edge, a 512×64-bit K^T/V memory feeding the B edge, and a 2 KiB
  UART buffer. Weights and position table start from `weights/*.mem` images
  baked into the bitstream and can be replaced at run time with LOAD/LOADPOS.
* **Layer-norm unit**: 8 rows in parallel lanes, one multiplier for c² and c/√V,
  1/√ from a 768-entry ROM after an even-shift normalisation.
* **Tile writeback** applies saturation, ReLU and the residual add (from the
  activation memory or the position table) one value per cycle.
* **Softmax unit**: per-row max, `EXP_LUT` ROM, 33-cycle restoring divider and
  the remainder fix-up that makes each row sum to exactly 65536.
* **UART protocol** at 1 Mbaud (27 clocks per bit) over the on-board BL616 USB bridge, framed
  `A5 5A len16 payload xor`. Commands: ECHO, LOAD, ATTN, INFO, INFER, LOADPOS.

A full inference takes 79,917 cycles (2.96 ms at 27 MHz), measured on chip;
an `INFER` round trip over USB is about 17 ms at 1 Mbaud. Resources: 70% LUT4
plus 18% ALU, 41% FF, 38/46 BSRAM, 33/48 MULT18X18; Fmax ≈ 34 MHz.

| LED (active low) | Meaning |
|---|---|
| 0 | heartbeat |
| 1 | request in progress |
| 2 | attention core computing |
| 3 | UART byte received |
| 4 | checksum error seen (sticky) |
| 5 | in reset (S1 held) |

## Layout

* `rtl/tang_nano_20k_top.sv` — board wrapper: power-on reset, S1 reset, LEDs.
* `rtl/tpu_top.sv` — command decoder, memories, request/response sequencing.
* `rtl/tpu_core.sv` — model sequencer, operand skew, tile writeback, memory map.
* `rtl/ln_unit.sv`, `rtl/rsqrt_rom.mem` — integer layer norm.
* `rtl/systolic_8x4.sv`, `rtl/pe.sv` — the MAC array.
* `rtl/softmax_unit.sv`, `rtl/exp_rom.sv` — softmax (ROM generated from the oracle).
* `rtl/packet_parser.sv`, `rtl/uart_rx.sv`, `rtl/uart_tx.sv`, `rtl/sdp_ram.sv`.
* `rtl/uart_echo_top.sv` — minimal bring-up top (byte echo).
* `constraints/tang_nano_20k.cst` — pins from Sipeed's official examples.
* `sim/` — array, UART and full end-to-end testbenches.
* `python/q8_attention.py` — bit-exact oracle and frame codec.
* `python/host.py` — `TpuLink` client and `selftest` CLI.
* `python/sim_vectors.py` — builds the end-to-end testbench script from the oracle.
* `python/model_export.py`, `train.py`, `dataset.py`, `app.py` — PC-side model and UI.

## Quick start

See [docs/BUILD.md](docs/BUILD.md) for the full flow. With OSS CAD Suite installed:

```sh
make test                         # Python oracle tests
make sim TOOLS=/path/to/oss-cad-suite
make bitstream TOOLS=/path/to/oss-cad-suite
make prog TOOLS=/path/to/oss-cad-suite      # SRAM; use 'flash' to persist
python python/host.py COM4 selftest         # random weights + inputs vs the oracle
```
