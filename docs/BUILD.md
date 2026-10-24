# Build and validation

## Tools

The open flow uses [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build)
(yosys, nextpnr-himbaechel with the Gowin backend, apicula's `gowin_pack`,
openFPGALoader, Icarus Verilog). On Windows, extract it and either add `bin`
and `lib` to `PATH` or pass `TOOLS=<install dir>` to make. The Makefile needs
GNU make and a POSIX shell. Git Bash ships without `make`; install it (for
example `winget install ezwinports.make`) or use MSYS2/WSL, or run the
yosys / nextpnr-himbaechel / gowin_pack lines from the `build/tpu.fs`
recipe by hand.

## Python oracle

```sh
make test
make generate    # regenerates rtl/exp_rom.sv from EXP_LUT
```

`test_exp_rom_matches_generator` fails if `rtl/exp_rom.sv` drifts from the
oracle's table.

## RTL simulation

```sh
make sim
```

* `sim-systolic` — random 8×K·K×4 tiles, both product modes, every PE.
* `sim-uart` — byte echo through `uart_echo_top`.
* `sim-top` — the end-to-end check. `python/sim_vectors.py` builds a byte
  script (INFO, ECHO, a corrupted frame, four weight loads, two attention
  layers including a saturating one, a bad-length request) and
  `tb_tpu_top.sv` drives it through the real UART at 8 clocks per bit,
  comparing every response byte with `attention_q8_8`. About one minute in
  Icarus.

## Bitstream

```sh
make bitstream
```

This runs `synth_gowin -family gw2a`, then nextpnr-himbaechel
(`--device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C`) at 27 MHz, then
`gowin_pack`, producing `build/tpu.fs`. Check `build/nextpnr.log` for
utilisation and the achieved Fmax. The design runs straight from the 27 MHz
oscillator with no PLL; the UART is the bottleneck, not the array.

## Programming

The BL616 on the board enumerates as an FTDI-compatible dual-channel device
(0403:6010). Interface A is JTAG and interface B is the UART.

* **Windows + openFPGALoader**: use [Zadig](https://zadig.akeo.ie) to put
  WinUSB on interface A only ("USB Serial Converter A" / "JTAG Debugger
  (Interface 0)"). Leave interface B on the FTDI VCP driver so the UART
  stays a COM port.
* **Gowin Programmer** works with the stock driver instead. Load
  `build/tpu.fs` with "SRAM Program" or "embFlash Erase, Program".

```sh
make prog     # SRAM, lost at power-off
make flash    # on-board flash, persists
```

## On-board test

```sh
python -m pip install pyserial
python python/host.py COM4 info
python python/host.py COM4 selftest --runs 4
```

`selftest` checks ECHO, runs the baked model against `tpu_model.forward`,
then loads a random model and compares INFER and ATTN outputs element by
element. `python python/evaluate.py --port COM4` scores the baked model on
the held-out sets on the FPGA.
On Windows the UART is the higher-numbered of the two COM ports the board
creates.

## Protocol

Frames are `A5 5A len_lo len_hi payload xor`. Request payloads are
`op arg data`; responses are `op status data`.

| op | arg | data | response |
|---|---|---|---|
| `01` ECHO | any | ≤ 2048 bytes | `01 00` + data |
| `02` LOAD | slot: layer·6 + (q,k,v,o,w1,w2); 12 embed; 13 output | lsh, then int8 W[k][c] row-major (64×64, 16×64, 64×16) | `02 00` |
| `03` ATTN | layer | 16×64 int16 LE X | `03 00` + 16×64 int16 LE attention output |
| `04` INFO | — | — | `04 00 'TPU' 02` + cycles (u32 LE) |
| `05` INFER | 0 | 16×16 int16 LE features | `05 00` class, cycles (u32 LE), 16×16 int16 LE logits |
| `06` LOADPOS | 0 | 16×64 int16 LE position table | `06 00` |
| bad checksum | | | `EE 01` |

LOAD writes weights while bytes arrive. If it is answered with `EE`, resend it.
A weight's Q8.8 value is `int8 << lsh`; `tpu_model.quantize` picks the
smallest shift per matrix. `cycles` is the core's compute time for the last
ATTN or INFER.

## Changing the baked model

```sh
python python/train.py        # writes weights/ in both formats
make export                   # or: python python/tpu_model.py (convert existing weights/)
make bitstream && make flash
```

`weights/tpu_weights.mem`, `tpu_lsh.mem` and `tpu_position.mem` become BSRAM
initial values. To try a model without rebuilding, send it at run time with
`TpuLink.load_model`; it lasts until reset.
