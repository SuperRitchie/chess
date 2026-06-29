# ml/train_safe_export.py
"""
run training with a local TensorFlow.js Layers export shim
"""
import os
import pathlib
import sys
import types

# GitHub's CPU runners can expose partial CUDA libraries. Force CPU-only TensorFlow
# before importing TensorFlow through train.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")

from tfjs_layers_export import export_keras_layers_model


def _save_keras_model(model, path):
    return export_keras_layers_model(model, pathlib.Path(path))


# train.py imports `tensorflowjs` at export time. Provide the small part it uses
# locally so the workflow does not need the tensorflowjs package or its
# TensorFlow Decision Forests dependency.
sys.modules["tensorflowjs"] = types.SimpleNamespace(
    converters=types.SimpleNamespace(save_keras_model=_save_keras_model)
)

from train import main


if __name__ == "__main__":
    main()
