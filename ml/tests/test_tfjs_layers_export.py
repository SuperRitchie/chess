import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf


ML_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from tfjs_layers_export import WEIGHTS_FILENAME, export_keras_layers_model


class TFJSLayersExportTests(unittest.TestCase):
    def test_float16_export_removes_stale_shards(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(2),
        ])
        model(np.zeros((1, 2), dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            (out_dir / "group1-shard1of3.bin").write_bytes(b"stale")

            model_json_path = export_keras_layers_model(model, out_dir)
            model_json = json.loads(model_json_path.read_text(encoding="utf-8"))
            weights = model_json["weightsManifest"][0]["weights"]

            self.assertEqual([WEIGHTS_FILENAME], model_json["weightsManifest"][0]["paths"])
            self.assertTrue(all(weight["dtype"] == "float32" for weight in weights))
            self.assertTrue(
                all(weight["quantization"] == {"dtype": "float16"} for weight in weights)
            )
            self.assertFalse((out_dir / "group1-shard1of3.bin").exists())

            scalar_count = sum(np.prod(weight["shape"]) for weight in weights)
            self.assertEqual(2 * scalar_count, (out_dir / WEIGHTS_FILENAME).stat().st_size)


if __name__ == "__main__":
    unittest.main()
