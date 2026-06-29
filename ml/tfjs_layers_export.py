# ml/tfjs_layers_export.py
"""
Small TensorFlow.js Layers exporter for this repo's Keras model.

This avoids the tensorflowjs Python converter dependency, which pulls in
TensorFlow Decision Forests even though this project only exports a standard
Keras Conv/Dense neural network.
"""
import json
import pathlib

import numpy as np
import tensorflow as tf


WEIGHTS_FILENAME = "group1-shard1of1.bin"


def keras_version() -> str:
    return getattr(tf.keras, "__version__", "unknown")


def weight_name(layer: tf.keras.layers.Layer, weight) -> str:
    name = getattr(weight, "path", None) or getattr(weight, "name", "")
    name = name.split(":", 1)[0]
    if "/" not in name:
        name = f"{layer.name}/{name}"
    return name


def float32_bytes(array: np.ndarray) -> bytes:
    array = np.asarray(array)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array).tobytes(order="C")


def export_keras_layers_model(model: tf.keras.Model, out_dir: pathlib.Path) -> pathlib.Path:
    """Write a TFJS Layers model.json and one weight shard for model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_manifest = []
    weight_chunks = []
    for layer in model.layers:
        for weight in layer.weights:
            array = weight.numpy()
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(f"Unsupported non-floating weight {weight.name}: {array.dtype}")
            weights_manifest.append({
                "name": weight_name(layer, weight),
                "shape": list(array.shape),
                "dtype": "float32",
            })
            weight_chunks.append(float32_bytes(array))

    (out_dir / WEIGHTS_FILENAME).write_bytes(b"".join(weight_chunks))

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_version()}",
        "convertedBy": "ml/tfjs_layers_export.py",
        "modelTopology": {
            "keras_version": keras_version(),
            "backend": "tensorflow",
            "model_config": json.loads(model.to_json()),
        },
        "weightsManifest": [
            {
                "paths": [WEIGHTS_FILENAME],
                "weights": weights_manifest,
            }
        ],
    }

    model_json_path = out_dir / "model.json"
    model_json_path.write_text(json.dumps(model_json), encoding="utf-8")

    if not weights_manifest:
        raise ValueError("TFJS export produced no weights")
    if not (out_dir / WEIGHTS_FILENAME).exists():
        raise FileNotFoundError(f"TFJS weight shard missing: {WEIGHTS_FILENAME}")

    return model_json_path
