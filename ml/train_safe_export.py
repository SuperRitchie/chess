# ml/train_safe_export.py
"""
run training with a guarded TensorFlow.js export import
"""

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
