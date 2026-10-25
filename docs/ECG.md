# ECG atrial fibrillation: data, features and model

Status: trained and evaluated in software with patient-grouped
cross-validation, and run on the board (model loaded at runtime, beat
features computed on the PC). No ECG sensor is connected yet; everything
here comes from recorded data.

```sh
python python/ecg_dataset.py                      # data/ecg_af_sequences.npz, about 35 s
python python/train_ecg.py --fold 0               # one CV fold (0-4); run all five
python python/train_ecg.py --summary              # baseline vs transformer, per fold and patient
python python/train_ecg.py --final                # weights/ecg_af/ (+ config.json)
python python/ecg_host.py COM4 --records 08455    # on the board, checked against the PC
```

## Data

MIT-BIH Atrial Fibrillation Database ([PhysioNet afdb
1.0.0](https://physionet.org/content/afdb/1.0.0/)): 25 ten-hour two-lead
recordings at 250 Hz, one subject each, with manually prepared rhythm
annotations. 00735 and 03665 have no signal files, leaving 23 subjects.
Download the `.hea/.dat/.atr/.qrs` files into `data/afdb/` (the S3 mirror
`physionet-open/afdb/1.0.0` is fastest; about 600 MB).

Target: **AF versus non-AF**. A rhythm annotation starts an interval that
lasts until the next one. `(AFIB` is AF; `(N` is non-AF, which in this
database means "all other rhythms", not necessarily healthy sinus rhythm.
`(AFL` (flutter) and `(J` (junctional) intervals are excluded from the
binary task for now, as is anything before a record's first annotation.

## Preprocessing and features (`python/ecg_dataset.py`)

The same functions serve training and inference.

- Lead ECG1 only (a single lead, as a wearable gives). Missing samples are
  set to 0 and flagged; so are clipped samples (|x| >= 10 mV, the 12-bit
  ADC limit) and flat-line stretches (lead off).
- 0.5-40 Hz zero-phase band-pass removes baseline drift and mains noise.
- R peaks: Pan-Tompkins-style detector (5-15 Hz band, derivative, square,
  150 ms integration, adaptive threshold from each 10-s block, 250 ms
  refractory), moved to the largest deflection of either polarity.
  Against the database's own `.qrs` beat annotations it reaches 94.5%
  sensitivity and 93.6% positive predictivity averaged over records
  (above 92% on 21 of 23). Record 07162 scores about 16% on both, but lead
  2 there gives the same beat count as the reference with random offsets,
  so its `.qrs` file looks misaligned; lead 1 also misses about 14% of its
  beats. 07859 reaches 86% sensitivity.
- A sequence is **16 consecutive beats**, all inside one rhythm interval
  together with the 8 beats before them (the look-back of the rolling
  statistics); transitions are excluded. Sequences start every 8 beats.
  23 subjects give 138,720 sequences, 61,917 of them AF.

16 features per beat (the proposed design, to be compared experimentally):

| Group | Features |
|---|---|
| Rhythm (4) | previous RR interval, its change from the last one, SD and RMSSD of the last 8 RR intervals |
| Shape (8) | DCT coefficients 1-8 of the beat window (-250 to +350 ms around R, so the P wave is inside), mean-removed and scaled to unit variance |
| Size (2) | QRS width (samples above half the R amplitude within +-60 ms), mean absolute area over R amplitude |
| Quality (2) | correlation with the median of the previous 8 beats, fraction of missing/clipped/flat samples |

Each feature is normalised with the median and 1.4826 x MAD of the
**training patients only**, clipped to +-6 and stored as Q8.8 (x 256), the
core's input format.

## Evaluation

Patient-grouped 5-fold cross-validation: every sequence of a subject stays
in one fold (folds deal subjects sorted by AF share, so each fold mixes AF
burdens). For test fold k, fold k+1 is the validation fold, which alone
picks the checkpoint and the decision threshold (maximum balanced
accuracy); the other three folds train. Reported per sequence: sensitivity,
specificity, balanced accuracy, average precision (PR-AUC), and per subject.

- **Baseline**: logistic regression on four interval statistics of the 16
  RR intervals (coefficient of variation, RMSSD / mean RR, mean |dRR| /
  mean RR, fraction of |dRR| > 50 ms), class-weighted.
- **Transformer**: the on-chip model (2 layers, width 64, about 52,000 int8
  parameters; `HeartNet` from `train_heart.py`), every (subject, label)
  group sampled equally, a float stage then quantisation-aware steps on the
  exact integer path. Its score is pooled[1] - pooled[0].

## Results (test folds, mean over 5 folds)

| Model | Sensitivity | Specificity | Balanced | PR-AUC |
|---|---|---|---|---|
| Interval baseline | 91.9% | 86.6% | 89.2% | 0.899 |
| Transformer, all 16 features, 3,000 + 1,500 steps | 86.7% | 82.0% | 84.3% | 0.883 |

The first transformer does **worse** than the interval baseline. Its
training loss falls to about 0 while some held-out subjects fail badly
(specificity 0.2% on 08215, 10% on 04908): with 14 training subjects it
memorises subject-specific beat shapes. Next to try: the rhythm features
only (`--feature-set rhythm`) and shorter training; early validation runs of
both looked better but were stopped before their test folds were scored.

## On the board

`ecg_host.py` loads `weights/ecg_af/` into the board at runtime (0.5 s,
LOAD; no bitstream change) and sends sequences as INFER requests. On 300
sequences from three subjects: **300/300 identical to the PC reference**,
80,941 cycles = 3.00 ms per inference. The baked heart model returns after
the next bitstream load or power cycle.

## Next

- Multiclass version on the PhysioNet 2017 Challenge (normal, AF, other
  rhythm, noisy), keeping the classes separate.
- Flutter and junctional rhythm as their own classes.
- Beat detection on the chip for a standalone ECG device.
