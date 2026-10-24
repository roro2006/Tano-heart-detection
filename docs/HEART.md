# Heart-sound screening: data, features and model

Status: the feature spec and a first model exist in Python, bit-exact with the
on-chip format. The audio front end is not in RTL yet, and the board has not
run this model. Results below are a baseline, not a screening-grade model.

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
