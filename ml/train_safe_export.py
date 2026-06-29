# ml/train_safe_export.py
"""
run training with a guarded TensorFlow.js export import
"""
import logging
import os
import warnings

# GitHub's CPU runners can expose partial CUDA libraries. Force CPU-only TensorFlow
# before importing train.py so cuFFT/cuDNN/cuBLAS/cuInit noise is avoided.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", message=".*HDF5 file format is considered legacy.*")
logging.getLogger("absl").setLevel(logging.ERROR)

try:
    from google.protobuf import runtime_version

    def _skip_unused_tfdf_protobuf_check(*args, **kwargs):
        return None

    runtime_version.ValidateProtobufRuntimeVersion = _skip_unused_tfdf_protobuf_check
except Exception:
    pass

from train import main


if __name__ == "__main__":
    main()
