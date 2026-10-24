import random
import struct
import tempfile
import unittest
from pathlib import Path

import dataset
import tpu_model
from evaluate import score
from host import OP_INFER, OP_LOAD, infer_response, load_request, random_int8, random_model
from model_export import embed_grid, layer_norm_q8_8, load_model
from q8_attention import unpack_frame
from tpu_model import Int8Matrix, layer_norm_int, quantize

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


class LayerNormTests(unittest.TestCase):
    def test_close_to_float_layer_norm(self):
        # Flooring the mean costs up to 1 LSB of c, which tiny rows (|r| ~ 30,
        # about 0.1 real) amplify to ~12 output LSB; large rows stay within 1%.
        rng = random.Random(1)
        for scale, slack in ((30, 24), (3000, 4), (30000, 4)):
            row = [rng.randint(-scale, scale) for _ in range(64)]
            for got, want in zip(layer_norm_int([row])[0], layer_norm_q8_8([row])[0]):
                self.assertLessEqual(abs(got - want), max(slack, abs(want) // 100))

    def test_constant_row_is_zero(self):
        self.assertEqual(layer_norm_int([[1234] * 64]), [[0] * 64])

    def test_single_step_row(self):
        # c = 1 in one place: 2048 / sqrt(1), through the table's 0.1% error.
        self.assertEqual(layer_norm_int([[5] * 63 + [6]])[0][-1], 2046)


class ModelTests(unittest.TestCase):
    def test_identity_embedding_matches_grid_embedding(self):
        grid = dataset.shifted(dataset.patterns()["box"], 1, 1)
        model = tpu_model.TpuModel(tpu_model.identity_embed(), (), quantize([[0.0] * 16] * 64),
                                   tpu_model.sinusoidal_position())
        self.assertEqual(tpu_model.embed(model, tpu_model.grid_features(grid)), embed_grid(grid))

    def test_quantize_picks_smallest_shift(self):
        matrix = quantize([[1.0, -0.5]])          # 256 and -128 in Q8.8
        self.assertEqual((matrix.lsh, matrix.values), (2, ((64, -32),)))
        self.assertEqual(matrix.q8_8(), [[256, -128]])

    def test_export_load_round_trip(self):
        model = random_model(random.Random(2))
        with tempfile.TemporaryDirectory() as tmp:
            tpu_model.export(Path(tmp), model)
            self.assertEqual(tpu_model.load(Path(tmp)), model)
            self.assertEqual(len((Path(tmp) / "tpu_weights.mem").read_text().split()), tpu_model.W_DEPTH)

    def test_weight_words_layout(self):
        model = random_model(random.Random(3))
        words = tpu_model.weight_words(model)
        # Slot 7 (layer 1, k), W[k=5][c=9]: word 7*1024 + (9 // 4)*64 + 5, lane 1.
        self.assertEqual((words[7 * 1024 + 2 * 64 + 5] >> 8) & 0xFF,
                         model.slot(7).values[5][9] & 0xFF)
        # Output (64x16), W[k=63][c=15]: word 12544 + 3*64 + 63, lane 3.
        self.assertEqual(words[12544 + 3 * 64 + 63] >> 24, model.output.values[63][15] & 0xFF)

    def test_load_frame_layout(self):
        matrix = random_int8(random.Random(4), 16, 64, lsh=3)
        payload = unpack_frame(load_request(tpu_model.SLOT_EMBED, matrix))
        self.assertEqual(payload[:3], bytes([OP_LOAD, 12, 3]))
        self.assertEqual(len(payload), 3 + 16 * 64)
        self.assertEqual(struct.unpack("b", payload[3 + 64 + 2:3 + 64 + 3])[0], matrix.values[1][2])

    def test_infer_response_layout(self):
        logits = [[0] * 16 for _ in range(16)]
        logits[3][7] = 5
        payload = unpack_frame(infer_response(logits, 0x01020304))
        self.assertEqual(payload[:7], bytes([OP_INFER, 0, 7, 4, 3, 2, 1]))
        self.assertEqual(len(payload), 7 + 512)

    def test_int8_range_is_enforced(self):
        with self.assertRaises(ValueError):
            Int8Matrix(((128,),), 0)


@unittest.skipUnless((WEIGHTS / tpu_model.MANIFEST).exists(), "no exported on-chip model")
class TrainedModelTests(unittest.TestCase):
    def test_baked_model_reconstructs_noisy_patterns(self):
        model = tpu_model.load(WEIGHTS)
        records = dataset.hard_set(8, seed=99, drop=0.05, salt=0.0)
        predictions = [tpu_model.pixels(tpu_model.forward(model, tpu_model.grid_features(r["pixels"])))
                       for r in records]
        metrics = score(records, predictions)
        self.assertGreater(metrics["pixel_accuracy"], 0.98)
        self.assertEqual(metrics["class_accuracy"], 1.0)

    def test_baked_model_matches_conversion_of_trained_weights(self):
        self.assertEqual(tpu_model.load(WEIGHTS), tpu_model.from_quant_model(load_model(WEIGHTS)))


if __name__ == "__main__":
    unittest.main()
