# Weights

Attention weights are no longer baked into the bitstream. The host loads them
over UART at run time (`TpuLink.load_weights`, or a `LOAD` frame per matrix),
so retraining needs no resynthesis. They live in BSRAM until power-off.

Each matrix is 64×64 signed Q8.8, sent as little-endian int16 in row-major
order W[k][c], the same orientation `attention_q8_8(x, wq, wk, wv, wo)` uses:
`x @ W`. In `model_export.py`, `export_layer` still writes the `$readmemh`-style
`layer{0,1}_{q,k,v,o}.mem` files (one four-hex-digit word per line, row-major),
and `host.read_matrix` reads them back for loading.

On the FPGA, word `layer*4096 + m*1024 + cg*64 + k` holds
`W_m[k][4cg .. 4cg+3]`, which is one B-edge column group of the systolic array.

## Trained model

`python python/train.py` trains the two-layer model that
`model_export.full_inference` runs and writes it here with
`model_export.export_model`:

| File | Shape | Where it runs |
|---|---|---|
| `layer{0,1}_{q,k,v,o}.mem` | 64×64 | FPGA (sent with `LOAD`) |
| `layer{0,1}_w1.mem`, `layer{0,1}_w2.mem` | 64×H, H×64 | PC (feed-forward) |
| `output.mem` | 64×16 | PC (output projection, sigmoid) |
| `model.json` | | manifest: shapes, training settings, integer-path metrics |

`model_export.load_model(Path("weights"))` returns a `QuantModel`. Load its
`attention[layer]` dicts with `TpuLink.load_weights(layer, ...)` after every
power-up, then call
`full_inference(grid, m.attention, m.feed_forward, m.output, attention_transport=tpu.attention)`.

The task is denoising: the input is a noisy 16×16 grid from `dataset.py` and
the target is the clean, shifted pattern. Each output is a Q8.8 sigmoid value,
and a pixel counts as lit when it is ≥ 128. `python python/evaluate.py [--port COM4]`
reports accuracy on the held-out records through the bit-exact integer path,
and with `--port` also checks the FPGA-backed outputs against the oracle.

`softmax_exp.mem` is the Q0.16 `exp(-n/16)` table (five hex digits, since
entry 0 is 65536) and is only a reference copy. The RTL uses the generated
`rtl/exp_rom.sv`.
