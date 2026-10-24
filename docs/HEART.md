# Heart-sound screening: data, features and model

Status: the feature spec, a first model and the FPGA audio front end exist.
The front end streams raw 2 kHz samples over USB (AUDIO) into the on-chip
filterbank and transformer, bit-exact with the Python spec in simulation;
the board build fits and meets timing but has not run on hardware yet. A
microphone input is still to come. Results below are a baseline, not a
screening-grade model.

## On the FPGA (`rtl/audio_frontend.sv`)

The board streams audio through the same pipeline as `heart_features.py`:
16 biquads on three multipliers (three cycles per band, about 50 cycles per
sample), per-band energies and log2 every 256 samples into a 16-frame ring,
and every 8 frames after warm-up the mean-removed 16x16 window goes straight
into the core's feature region and the whole model runs. The board keeps the
number of windows and the sum of (pooled[1] − pooled[0]) since the last
stream reset: positive means "abnormal: refer", the same rule as training.

```sh
python python/heart_host.py COM4 a0001          # one recording, checked against the PC
python python/heart_host.py COM4 --split test   # sensitivity/specificity measured on the board
make sim-audio                                  # a real recording through the RTL, bit-exact
```

The bitstream bakes `weights/heart/` (`rtl/tang_nano_20k_top.sv`), so the
board boots ready to screen. Resources with the front end: 76% LUT4 + 20%
ALU, 47% FF, 40/46 BSRAM, 36/48 MULT18X18, Fmax ≈ 34 MHz at a 27 MHz clock.
To fit, the per-band filter state lives in a small RAM and the exp, log2 and
coefficient tables have registered outputs so they map to block RAM.

## Data

PhysioNet/CinC Challenge 2016 training set (Open Data Commons Attribution
1.0): 3,240 recordings, 665 abnormal, 2 kHz 16-bit mono, 5–122 s each
(20.2 h). Download `training.zip` (190 MB) into `data/physionet2016/` (the
official S3 mirror `physionet-open` is much faster than physionet.org), then:

```sh
python python/heart_dataset.py     # data/heart_windows.pt, about 1.5 minutes
python python/train_heart.py       # weights/heart/, about 6 minutes on a CPU
```

The six source databases (a–f) differ in equipment, population and abnormal
rate (a: 71% abnormal, e: 8.5%). The split is by recording, stratified by
database and label (70/15/15). The public files do not identify subjects, so
some subject overlap between splits is possible.

## Features (`python/heart_features.py`, integer, FPGA-ready)

- 16 bandpass biquads, 25–800 Hz log-spaced, Q = 3, Q2.16 coefficients,
  18-bit state with 2 fractional bits; 3 multiplies per band per sample.
- Frame = 256 samples (128 ms): per band, the sum of |y|.
- log2 in Q8.8 from the MSB position plus a 256-entry fraction table.
- Window = 16 frames (2.048 s); subtracting the window mean removes the
  microphone gain (a 2x gain change moves a feature by about 1 LSB median).

The window is exactly the model input: 16 tokens (frames) × 16 features
(bands).

## Model and training (`python/train_heart.py`)

The on-chip transformer (`tpu_model.py`) with a learned embedding and position
table, classes 0 = normal and 1 = abnormal. Training is a float stage, then
quantisation-aware steps on the exact integer path. Batches sample every
(database, label) group equally, so a recording's source carries no label
information, and checkpoints are chosen by the mean of per-database
balanced accuracies on validation. A recording is called abnormal when the
mean of (pooled[1] − pooled[0]) over its 2-s windows is positive.

## Results (test split, 484 recordings)

| | Sensitivity | Specificity | Balanced |
|---|---|---|---|
| all recordings | 71.7% | 89.9% | 80.8% |
| mean over databases | | | 72.9% |
| database a (61) | 59.1% | 64.7% | |
| database b (73) | 66.7% | 63.8% | |
| database e (321) | 100% | 97.3% | |
| database f (17) | 40.0% | 83.3% | |

Databases c and d have 4 and 8 test recordings; their numbers are noise.

Measured on the board (bitstream `da32570`, Tang Nano 20K, 2026-10-24,
`python python/heart_host.py COM4 --split test`): all 484 test recordings in
1,597 s, 484/484 identical to the PC reference, sensitivity 71.7%,
specificity 89.9%, balanced 80.8%. The hardware reproduces the PC model
exactly, so the numbers above are the board's numbers. Note that
`host.py selftest` loads a random model and overwrites the baked weights;
reload the bitstream before running `heart_host.py` after a selftest.

Read these with care. The pooled number is dominated by database e, which
the model separates almost perfectly; that may be genuine or another
recording-condition confound. On the other databases the model is near
60–65% balanced. A first version trained with plain class balancing scored
87.8% pooled but only by learning which database a recording came from
(it called almost every database-a recording abnormal), which is why the
sampling above is per database. A new stethoscope microphone is effectively
a new database, so the per-database numbers are the relevant ones.

For context, the best 2016 challenge entries reported around 86% on the
challenge's hidden test set using heartbeat segmentation (S1/S2/systole/
diastole) and much larger feature sets.

## Likely improvements

- Heart-cycle segmentation, or features aligned to S1/S2.
- Longer context per decision (several windows combined on chip).
- Augmentation: gain, background noise, small pitch/tempo changes.
- Leave-one-database-out evaluation to measure transfer to unseen hardware.
- Recording our own stethoscope microphone and fine-tuning on it.
