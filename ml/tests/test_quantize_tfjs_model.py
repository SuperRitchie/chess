import json
import pathlib
import struct
import sys
import tempfile
import unittest


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from quantize_tfjs_model import quantize_model


class QuantizeTFJSModelTests(unittest.TestCase):
    def test_quantizes_float32_weights_and_removes_stale_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp)
            model_path = model_dir / "model.json"
            shard_path = model_dir / "group1-shard1of1.bin"
            stale_path = model_dir / "group1-shard1of2.bin"
            model_path.write_text(json.dumps({
                "weightsManifest": [{
                    "paths": [shard_path.name],
                    "weights": [
                        {"name": "dense/kernel", "shape": [2], "dtype": "float32"},
                        {"name": "dense/bias", "shape": [1], "dtype": "float32"},
                    ],
                }],
            }), encoding="utf-8")
            shard_path.write_bytes(struct.pack("<fff", 0.25, -1.5, 3.125))
            stale_path.write_bytes(b"stale")

            sizes = quantize_model(model_path)
            updated = json.loads(model_path.read_text(encoding="utf-8"))
            weights = updated["weightsManifest"][0]["weights"]

            self.assertEqual((12, 6), sizes)
            self.assertEqual((0.25, -1.5, 3.125), struct.unpack("<eee", shard_path.read_bytes()))
            self.assertTrue(
                all(weight["quantization"] == {"dtype": "float16"} for weight in weights)
            )
            self.assertFalse(stale_path.exists())
            self.assertEqual((6, 6), quantize_model(model_path))

    def test_rejects_a_truncated_weight_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp)
            model_path = model_dir / "model.json"
            model_path.write_text(json.dumps({
                "weightsManifest": [{
                    "paths": ["group1-shard1of1.bin"],
                    "weights": [
                        {"name": "dense/kernel", "shape": [2], "dtype": "float32"},
                    ],
                }],
            }), encoding="utf-8")
            (model_dir / "group1-shard1of1.bin").write_bytes(struct.pack("<f", 0.25))

            with self.assertRaisesRegex(ValueError, "has 4 bytes, expected 8"):
                quantize_model(model_path)


if __name__ == "__main__":
    unittest.main()
