"""run a reproducible accepted-model strength benchmark"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import tensorflow as tf

import train
import train_fixed_eval


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark(positions: int, searches: int) -> dict:
    np.random.seed(0)
    tf.keras.utils.set_random_seed(0)
    samples = train_fixed_eval.load_fixed_eval_set()
    X, P, V, PW, VW = train_fixed_eval.fixed_eval_arrays(samples)
    model = tf.keras.models.load_model(train.CHECKPOINT_MODEL, compile=False)

    fixed_metrics = train_fixed_eval.evaluate_fixed_model(
        model,
        X,
        P,
        V,
        PW,
        VW,
        "accepted fixed",
    )
    nn_metrics = train_fixed_eval.evaluate_nn_policy_alignment(
        model,
        samples,
        "accepted",
        limit=positions,
    )
    original_positions = train_fixed_eval.MCTS_EVAL_POSITIONS
    original_searches = train_fixed_eval.MCTS_EVAL_SEARCHES
    try:
        train_fixed_eval.MCTS_EVAL_POSITIONS = positions
        train_fixed_eval.MCTS_EVAL_SEARCHES = searches
        mcts_metrics = train_fixed_eval.evaluate_mcts_alignment(model, samples, "accepted")
    finally:
        train_fixed_eval.MCTS_EVAL_POSITIONS = original_positions
        train_fixed_eval.MCTS_EVAL_SEARCHES = original_searches

    artifacts = [
        train.CHECKPOINT_MODEL,
        train.OUT_DIR / "model.json",
        *sorted(train.OUT_DIR.glob("group*.bin")),
    ]
    return {
        "benchmark_version": 1,
        "fixed_holdout_sha256": sha256(train_fixed_eval.FIXED_EVAL_SET),
        "positions": positions,
        "searches": searches,
        "fixed_neural": fixed_metrics,
        "nn_to_stockfish": nn_metrics,
        "mcts_to_stockfish": mcts_metrics,
        "artifact_sha256": {
            str(path): sha256(path)
            for path in artifacts
            if path.exists()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="benchmark the accepted neural chess model")
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--searches", type=int, default=64)
    args = parser.parse_args()
    if args.positions <= 0 or args.searches <= 0:
        parser.error("positions and searches must be positive")
    print(json.dumps(benchmark(args.positions, args.searches), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
