import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import dataset
from evaluate import binarize, classify, score
from host import read_matrix
from model_export import QuantModel, export_model, full_inference, layer_norm_q8_8, load_model

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def random_model(rng: random.Random, hidden: int = 32) -> QuantModel:
    def matrix(rows, cols):
        return [[rng.randint(-32768, 32767) for _ in range(cols)] for _ in range(rows)]
    return QuantModel(
        attention=[{name: matrix(64, 64) for name in "qkvo"} for _ in range(2)],
        feed_forward=[(matrix(64, hidden), matrix(hidden, 64)) for _ in range(2)],
        output=matrix(64, 16),
    )


class ExportTests(unittest.TestCase):
    def test_export_load_round_trip(self):
        model = random_model(random.Random(5))
        with tempfile.TemporaryDirectory() as tmp:
            export_model(Path(tmp), model, {"note": "test"})
            self.assertEqual(load_model(Path(tmp)), model)

    def test_exported_matrices_read_back(self):
        model = random_model(random.Random(6))
        with tempfile.TemporaryDirectory() as tmp:
            export_model(Path(tmp), model)
            for layer in range(2):
                for name in "qkvo":
                    self.assertEqual(read_matrix(Path(tmp) / f"layer{layer}_{name}.mem", 64, 64),
                                     model.attention[layer][name])

    def test_layer_norm_is_unit_scale(self):
        # Real values 1..4 normalize to about -1.34, -0.45, 0.45, 1.34.
        self.assertEqual(layer_norm_q8_8([[256, 512, 768, 1024]]), [[-343, -114, 114, 343]])


class DatasetTests(unittest.TestCase):
    def test_build_records_have_clean_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset.build(Path(tmp) / "p.json", copies=5)
            records = dataset.load(Path(tmp) / "p.json")
        self.assertEqual(len(records), 20)
        for record in records:
            for noisy, clean in zip(record["pixels"], record["target"]):
                self.assertTrue(all(p <= t for p, t in zip(noisy, clean)))

    def test_split_holds_out_every_fifth(self):
        train, test = dataset.split(list(range(10)))
        self.assertEqual((train, test), ([1, 2, 3, 4, 6, 7, 8, 9], [0, 5]))

    def test_classify_templates(self):
        for name, image in dataset.patterns().items():
            expected = "cross" if name == "plus" else name
            self.assertEqual(classify(dataset.shifted(image, 1, -2)), expected)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class SimulatorTests(unittest.TestCase):
    def test_batched_integer_path_matches_full_inference(self):
        import train
        model = train.Denoiser(32, seed=1).quantized()
        records = dataset.hard_set(3)
        expected = [full_inference(r["pixels"], model.attention, model.feed_forward, model.output)
                    for r in records]
        self.assertEqual(train.int_forward(train.batch(records)[0], model).tolist(), expected)


@unittest.skipUnless((WEIGHTS / "model.json").exists(), "no trained weights")
class TrainedWeightTests(unittest.TestCase):
    def test_trained_model_reconstructs_noisy_patterns(self):
        model = load_model(WEIGHTS)
        records = dataset.hard_set(8, seed=99, drop=0.05, salt=0.0)
        predictions = [binarize(full_inference(r["pixels"], model.attention, model.feed_forward,
                                               model.output)) for r in records]
        metrics = score(records, predictions)
        self.assertGreater(metrics["pixel_accuracy"], 0.98)
        self.assertEqual(metrics["class_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
