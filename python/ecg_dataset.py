"""ECG atrial-fibrillation dataset: MIT-BIH AF Database -> 16-beat sequences.

    python python/ecg_dataset.py           # data/ecg_af_sequences.npz + data/ecg_af_manifest.json

Data: the MIT-BIH Atrial Fibrillation Database (PhysioNet afdb 1.0.0), 25
long-term two-lead recordings at 250 Hz, one subject each; 00735 and 03665
have annotations only and are skipped. Put the .hea/.dat/.atr/.qrs files in
data/afdb/ (the S3 mirror physionet-open/afdb/1.0.0 is fastest).

Target: AF versus non-AF. Rhythm annotations start an interval that lasts
until the next one. (AFIB -> 1, (N -> 0 ("all other rhythms", not
necessarily sinus), (AFL and (J are excluded from the binary task, as is
anything before a record's first rhythm annotation.

Pipeline, identical for training and inference (preprocess + beat_features):
lead ECG1 only (a single lead, as a wearable would give); missing samples
set to 0 and flagged; 0.5-40 Hz band-pass removes baseline drift; R peaks
from a Pan-Tompkins-style detector (5-15 Hz band, derivative, square, 150 ms
integration, adaptive threshold, 250 ms refractory), checked against the
database's own .qrs beat annotations. Each beat gets 16 features (FEATURES
below). A sequence is 16 consecutive beats whose rhythm label, and that of
the 8 beats before them (the rolling statistics' look-back), is one class;
sequences start every 8 beats. Nothing here is fitted to data: feature
normalisation is learned per training split in train_ecg.py.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import fft, signal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "afdb"
CACHE = ROOT / "data" / "ecg_af_sequences.npz"
MANIFEST = ROOT / "data" / "ecg_af_manifest.json"

FS = 250
BEATS, HOP, LOOKBACK = 16, 8, 8
PRE, POST = int(0.25 * FS), int(0.35 * FS)      # beat window: 250 ms before R to 350 ms after
RHYTHM = {"(AFIB": 1, "(N": 0}                  # (AFL, (J and anything else -> excluded (-1)
FEATURES = (
    "rr_prev_s", "rr_change_s", "rr_sd8_s", "rr_rmssd8_s",
    "dct1", "dct2", "dct3", "dct4", "dct5", "dct6", "dct7", "dct8",
    "qrs_width_ms", "area_over_peak", "template_corr8", "missing_or_clipped",
)
CLIP_MV = 10.0                                    # 12-bit ADC at 200/mV saturates at +-10.24 mV


def preprocess(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw single-lead ECG in mV -> (0.5-40 Hz filtered signal, bad-sample mask)."""
    bad = ~np.isfinite(raw)
    x = np.where(bad, 0.0, raw)
    bad |= np.abs(x) >= CLIP_MV
    # A flat line (lead off) shows as near-zero variation over half a second.
    w = FS // 2
    mean = np.convolve(x, np.ones(w) / w, "same")
    var = np.convolve((x - mean) ** 2, np.ones(w) / w, "same")
    bad |= var < 1e-6
    sos = signal.butter(2, [0.5, 40], "bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x), bad


def detect_r_peaks(x: np.ndarray) -> np.ndarray:
    """Pan-Tompkins-style R-peak detector on the filtered ECG; returns sample indices."""
    sos = signal.butter(2, [5, 15], "bandpass", fs=FS, output="sos")
    energy = np.diff(signal.sosfiltfilt(sos, x), prepend=0.0) ** 2
    win = int(0.15 * FS)
    mwi = np.convolve(energy, np.ones(win) / win, "same")
    # Adaptive threshold: a third of the 98th percentile of each 10-s block.
    block = 10 * FS
    starts = np.arange(0, len(mwi), block)
    levels = np.array([np.percentile(mwi[s:s + block], 98) for s in starts])
    threshold = 0.3 * np.interp(np.arange(len(mwi)), starts + block // 2, levels)
    peaks, _ = signal.find_peaks(mwi, height=threshold, distance=int(0.25 * FS))
    # Move each detection to the largest absolute deflection nearby (either polarity).
    lo, hi = int(0.12 * FS), int(0.08 * FS)
    refined = []
    for p in peaks:
        a, b = max(0, p - lo), min(len(x), p + hi)
        refined.append(a + int(np.argmax(np.abs(x[a:b]))))
    refined = np.unique(refined)
    keep = np.concatenate([[True], np.diff(refined) > int(0.2 * FS)])
    return refined[keep]


def detector_agreement(found: np.ndarray, reference: np.ndarray, tol: int = int(0.075 * FS)) -> dict:
    """Sensitivity and positive predictivity of `found` against reference beat samples."""
    idx = np.clip(np.searchsorted(reference, found), 1, len(reference) - 1)
    nearest = np.minimum(np.abs(found - reference[idx - 1]), np.abs(found - reference[idx]))
    true_found = int((nearest <= tol).sum())
    idx = np.clip(np.searchsorted(found, reference), 1, len(found) - 1)
    nearest = np.minimum(np.abs(reference - found[idx - 1]), np.abs(reference - found[idx]))
    matched = int((nearest <= tol).sum())
    return {"sensitivity": matched / len(reference), "ppv": true_found / len(found)}


def beat_features(x: np.ndarray, bad: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """[beats, 16] float features in FEATURES order. Beat 0 has no RR and is NaN."""
    n = len(peaks)
    out = np.full((n, len(FEATURES)), np.nan)
    rr = np.diff(peaks, prepend=np.nan) / FS
    out[:, 0] = rr
    out[:, 1] = np.diff(rr, prepend=np.nan)
    # Rolling SD and RMSSD over the last LOOKBACK intervals, ending at this beat.
    rr_win = np.lib.stride_tricks.sliding_window_view(rr, LOOKBACK)
    d_win = np.lib.stride_tricks.sliding_window_view(np.diff(rr, prepend=np.nan), LOOKBACK)
    out[LOOKBACK - 1:, 2] = rr_win.std(axis=1)
    out[LOOKBACK - 1:, 3] = np.sqrt(np.mean(d_win ** 2, axis=1))

    windows = np.zeros((n, PRE + POST))
    ok = (peaks >= PRE) & (peaks + POST <= len(x))
    for i in np.flatnonzero(ok):
        windows[i] = x[peaks[i] - PRE:peaks[i] + POST]
    # Shape: DCT coefficients 1-8 of the mean-removed, unit-variance window
    # (amplitude-free, so lead gain and electrode contact do not matter).
    centred = windows - windows.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    shape = np.divide(centred, scale, out=np.zeros_like(centred), where=scale > 0)
    out[:, 4:12] = fft.dct(shape, type=2, norm="ortho", axis=1)[:, 1:9]
    r_amp = np.abs(windows[:, PRE])
    near = np.abs(windows[:, PRE - int(0.06 * FS):PRE + int(0.06 * FS)])
    out[:, 12] = np.sum(near > 0.5 * r_amp[:, None], axis=1) * 1000 / FS
    out[:, 13] = np.divide(np.abs(centred).mean(axis=1), r_amp, out=np.zeros(n), where=r_amp > 0)
    for i in range(LOOKBACK, n):
        template = np.median(shape[i - LOOKBACK:i], axis=0)
        denom = np.linalg.norm(template) * np.linalg.norm(shape[i])
        out[i, 14] = float(template @ shape[i] / denom) if denom > 0 else 0.0
    bad_frac = np.zeros(n)
    for i in np.flatnonzero(ok):
        bad_frac[i] = bad[peaks[i] - PRE:peaks[i] + POST].mean()
    out[:, 15] = np.where(ok, bad_frac, 1.0)
    return out


def rhythm_labels(record: str, peaks: np.ndarray) -> np.ndarray:
    """Per-beat label: 1 AF, 0 non-AF, -1 excluded (flutter, junctional, before first label)."""
    import wfdb
    ann = wfdb.rdann(str(DATA / record), "atr")
    starts = np.array([s for s, note in zip(ann.sample, ann.aux_note) if note.startswith("(")])
    kinds = np.array([RHYTHM.get(note.strip("\x00"), -1) for note in ann.aux_note if note.startswith("(")])
    idx = np.searchsorted(starts, peaks, side="right") - 1
    return np.where(idx >= 0, kinds[np.clip(idx, 0, None)], -1)


def sequences(labels: np.ndarray, feats: np.ndarray) -> list[int]:
    """End-beat indices of valid 16-beat sequences (label constant over sequence + look-back)."""
    span = BEATS + LOOKBACK
    ends = []
    for end in range(span, len(labels), HOP):
        seg = labels[end - span + 1:end + 1]
        if seg[0] >= 0 and np.all(seg == seg[0]) and np.isfinite(feats[end - BEATS + 1:end + 1]).all():
            ends.append(end)
    return ends


def process(record: str) -> dict:
    import wfdb
    raw = wfdb.rdrecord(str(DATA / record), channels=[0]).p_signal[:, 0]
    x, bad = preprocess(raw)
    peaks = detect_r_peaks(x)
    feats = beat_features(x, bad, peaks)
    labels = rhythm_labels(record, peaks)
    ends = sequences(labels, feats)
    reference = wfdb.rdann(str(DATA / record), "qrs").sample
    X = np.stack([feats[e - BEATS + 1:e + 1] for e in ends]).astype(np.float32) if ends else \
        np.zeros((0, BEATS, len(FEATURES)), np.float32)
    rr = X[:, :, 0].copy()
    y = np.array([labels[e] for e in ends], dtype=np.int8)
    return {"record": record, "X": X, "rr": rr, "y": y,
            "start_sample": np.array([peaks[e - BEATS + 1] for e in ends], dtype=np.int64),
            "beats": len(peaks), "hours": len(raw) / FS / 3600,
            "beats_af": int((labels == 1).sum()), "beats_non_af": int((labels == 0).sum()),
            "beats_excluded": int((labels < 0).sum()),
            "detector_vs_qrs": detector_agreement(peaks, reference)}


def main() -> None:
    records = [r for r in (DATA / "RECORDS").read_text().split() if (DATA / f"{r}.dat").exists()] \
        if (DATA / "RECORDS").exists() else sorted(p.stem for p in DATA.glob("*.dat"))
    with Pool(min(len(records), 12)) as pool:
        results = pool.map(process, records)
    keep = [r for r in results if len(r["y"])]
    X = np.concatenate([r["X"] for r in keep])
    np.savez_compressed(
        CACHE, X=X, rr=np.concatenate([r["rr"] for r in keep]), y=np.concatenate([r["y"] for r in keep]),
        record=np.concatenate([np.full(len(r["y"]), i, np.int16) for i, r in enumerate(keep)]),
        start_sample=np.concatenate([r["start_sample"] for r in keep]),
        records=np.array([r["record"] for r in keep]), features=np.array(FEATURES))
    manifest = {
        "source": "MIT-BIH Atrial Fibrillation Database, PhysioNet afdb 1.0.0",
        "lead": "ECG1", "fs": FS, "beats_per_sequence": BEATS, "hop_beats": HOP, "lookback_beats": LOOKBACK,
        "beat_window_ms": [-PRE * 1000 // FS, POST * 1000 // FS], "band_pass_hz": [0.5, 40],
        "labels": {"1": "(AFIB", "0": "(N (all other rhythms, not necessarily sinus)"},
        "excluded": "(AFL, (J, beats before the first rhythm annotation, sequences crossing a rhythm change",
        "features": list(FEATURES),
        "records": {r["record"]: {k: r[k] for k in ("hours", "beats", "beats_af", "beats_non_af",
                                                     "beats_excluded", "detector_vs_qrs")}
                    | {"sequences_af": int((r["y"] == 1).sum()), "sequences_non_af": int((r["y"] == 0).sum())}
                    for r in results},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    se = np.mean([r["detector_vs_qrs"]["sensitivity"] for r in results])
    ppv = np.mean([r["detector_vs_qrs"]["ppv"] for r in results])
    print(f"{len(keep)} records, {len(X)} sequences ({int((np.concatenate([r['y'] for r in keep]) == 1).sum())} AF); "
          f"R detector vs .qrs: sensitivity {se:.3f}, PPV {ppv:.3f} (mean over records) -> {CACHE}")


if __name__ == "__main__":
    sys.exit(main())
