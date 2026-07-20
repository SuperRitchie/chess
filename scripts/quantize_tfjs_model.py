#!/usr/bin/env python3
import argparse
import json
import math
import os
import pathlib
import struct


def _scalar_count(weights: list[dict]) -> int:
    return sum(math.prod(weight["shape"]) for weight in weights)


def _write_atomic(path: pathlib.Path, data: bytes | str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(data, str):
        temporary.write_text(data, encoding="utf-8")
    else:
        temporary.write_bytes(data)
    os.replace(temporary, path)


def quantize_model(model_path: pathlib.Path) -> tuple[int, int]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model_dir = model_path.parent
    original_bytes = 0
    quantized_bytes = 0
    referenced_paths = set()

    for group in model.get("weightsManifest", []):
        paths = group.get("paths", [])
        weights = group.get("weights", [])
        if len(paths) != 1:
            raise ValueError("float16 optimizer requires one shard per weight group")

        shard_path = model_dir / paths[0]
        referenced_paths.add(shard_path.resolve())
        scalar_count = _scalar_count(weights)
        existing_quantization = {
            weight.get("quantization", {}).get("dtype") for weight in weights
        }

        if existing_quantization == {"float16"}:
            expected_size = scalar_count * 2
            actual_size = shard_path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"{shard_path.name} has {actual_size} bytes, expected {expected_size}"
                )
            original_bytes += actual_size
            quantized_bytes += actual_size
            continue

        if existing_quantization != {None}:
            raise ValueError(f"unsupported mixed quantization in {shard_path.name}")
        if any(weight.get("dtype") != "float32" for weight in weights):
            raise ValueError(f"unsupported weight dtype in {shard_path.name}")

        source = shard_path.read_bytes()
        expected_size = scalar_count * 4
        if len(source) != expected_size:
            raise ValueError(
                f"{shard_path.name} has {len(source)} bytes, expected {expected_size}"
            )

        output = bytearray(scalar_count * 2)
        for index, (value,) in enumerate(struct.iter_unpack("<f", source)):
            struct.pack_into("<e", output, index * 2, value)
        _write_atomic(shard_path, bytes(output))

        for weight in weights:
            weight["quantization"] = {"dtype": "float16"}
        original_bytes += len(source)
        quantized_bytes += len(output)

    _write_atomic(model_path, json.dumps(model))

    for shard_path in model_dir.glob("group*.bin"):
        if shard_path.resolve() not in referenced_paths:
            shard_path.unlink()

    return original_bytes, quantized_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize TFJS float32 weights to float16")
    parser.add_argument("model_json", type=pathlib.Path)
    args = parser.parse_args()
    original_bytes, quantized_bytes = quantize_model(args.model_json)
    print(f"optimized TFJS weights: {original_bytes} -> {quantized_bytes} bytes")


if __name__ == "__main__":
    main()
