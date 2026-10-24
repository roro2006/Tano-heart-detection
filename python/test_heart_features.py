import importlib.util
import math
import random
import unittest
from pathlib import Path

import heart_dataset
import heart_features as hf
import tpu_model

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DATA = (heart_dataset.DATA / "training-a" / "REFERENCE.csv").exists()


def tone(freq: float, amplitude: int = 8000, n: int = hf.WINDOW) -> list[int]:
    return [round(amplitude * math.sin(2 * math.pi * freq * i / hf.RATE)) for i in range(n)]


class FrontEndTests(unittest.TestCase):
    def test_coefficients_fit_18_bits_and_are_stable(self):
        for g, a1, a2 in hf.COEFFS:
            self.assertTrue(all(-(1 << 17) <= c < (1 << 17) for c in (g, a1, a2)))
            self.assertTrue(0 < a2 < (1 << hf.COEFF_BITS))

    def test_each_band_peaks_at_its_centre(self):
        for band, freq in enumerate(hf.band_centres()):
            energies = hf.frame_energies(tone(freq))[-1]
            self.assertEqual(energies.index(max(energies)), band, f"{freq:.0f} Hz")

    def test_full_scale_input_does_not_saturate_the_peak_band(self):
        energies = hf.frame_energies(tone(100.0, 32767))[-1]
        self.assertLess(max(energies), hf.FRAME * hf.Y_MAX)

    def test_log2_is_exact_on_powers_of_two(self):
        self.assertEqual([hf.log2_q8_8(1 << k) for k in (0, 1, 8, 20)], [0, 256, 2048, 5120])
        self.assertEqual(hf.log2_q8_8(0), 0)
        self.assertEqual(hf.log2_q8_8(3), 256 + round(256 * math.log2(1.5)))

    def test_window_mean_is_removed(self):
        logs = [[100 * t + b for b in range(16)] for t in range(16)]
        features = hf.window_features(logs)
        self.assertLess(abs(sum(map(sum, features))), 256)

    def test_gain_changes_features_little(self):
        rng = random.Random(1)
        signal = [round(3000 * math.sin(2 * math.pi * 60 * i / hf.RATE) + rng.gauss(0, 300))
                  for i in range(3 * hf.WINDOW)]
        half = [round(v / 2) for v in signal]
        diffs = sorted(abs(a - b) for wa, wb in zip(hf.recording_windows(signal), hf.recording_windows(half))
                       for ra, rb in zip(wa, wb) for a, b in zip(ra, rb))
        self.assertLessEqual(diffs[len(diffs) // 2], 4)

    def test_recording_windows_shape(self):
        windows = hf.recording_windows(tone(50.0, n=5 * hf.WINDOW))
        self.assertGreater(len(windows), 3)
        self.assertTrue(all(len(w) == 16 and all(len(r) == 16 for r in w) for w in windows))


class SplitTests(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self):
        items = [heart_dataset.Recording(f"{db}{i:04d}", db, int(i % 4 == 0), Path("x"))
                 for db in "abc" for i in range(200)]
        split = heart_dataset.assign_splits(items)
        self.assertEqual(set(split.values()), set(heart_dataset.SPLITS))
        for db in "abc":
            for label in (0, 1):
                names = [r.name for r in items if r.database == db and r.label == label]
                train = sum(split[n] == "train" for n in names) / len(names)
                self.assertAlmostEqual(train, 0.70, delta=0.03)

    @unittest.skipUnless(HAS_DATA, "PhysioNet 2016 data not downloaded")
    def test_physionet_recordings_load(self):
        items = heart_dataset.recordings()
        self.assertEqual(len(items), 3240)
        self.assertEqual(sum(r.label for r in items), 665)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class IntegerPathTests(unittest.TestCase):
    def test_batched_layer_norm_matches_spec(self):
        import torch
        import train_heart
        rng = random.Random(2)
        rows = [[rng.randint(-s, s) for _ in range(64)] for s in (3, 400, 32767)] + [[9] * 64]
        self.assertEqual(train_heart.int_layer_norm(torch.tensor(rows)).tolist(),
                         tpu_model.layer_norm_int(rows))

    def test_batched_forward_matches_spec(self):
        import torch
        import train_heart
        from host import random_model
        rng = random.Random(3)
        model = random_model(rng)
        features = [[rng.randint(-500, 500) for _ in range(16)] for _ in range(16)]
        logits = tpu_model.forward(model, features)
        pooled = train_heart.int_forward(model, torch.tensor([features]))[0].tolist()
        self.assertEqual(pooled, [sum(row[j] for row in logits) for j in range(16)])


if __name__ == "__main__":
    unittest.main()
