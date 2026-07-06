# ml/train_fixed_eval.py
"""
Training entrypoint with a persisted fixed evaluation set.

The normal train/validation split can change as new self-play samples are added.
This wrapper keeps training behavior the same, but gates checkpoint replacement on
one saved benchmark set so losses are comparable across nightly runs.
"""
import os
import random

import numpy as np
import tensorflow as tf

import train


FIXED_EVAL_SET = train.pathlib.Path("ml/data/fixed_eval_set.json")
FIXED_EVAL_SELF = int(os.environ.get("AZ_FIXED_EVAL_SELF", "512"))
FIXED_EVAL_STOCKFISH = int(os.environ.get("AZ_FIXED_EVAL_STOCKFISH", "512"))


def _valid_self_play_sample(item: dict) -> dict | None:
    fen = item.get("fen")
    if not fen:
        return None
    policy = item.get("policy")
    dense_policy = train.dense_policy_from_sparse(policy)
    if float(np.sum(dense_policy)) <= 0:
        return None
    try:
        z = float(item.get("z"))
    except (TypeError, ValueError):
        return None
    return {
        "source": "self_play",
        "fen": fen,
        "policy": policy,
        "z": float(np.clip(z, -1.0, 1.0)),
    }


def _valid_stockfish_sample(item: dict) -> dict | None:
    fen = item.get("fen")
    if not fen:
        return None
    try:
        cp = float(item.get("cp", 0.0))
    except (TypeError, ValueError):
        return None
    return {"source": "stockfish", "fen": fen, "cp": cp}


def _pick_deterministic(items: list[dict], count: int, salt: str) -> list[dict]:
    rng = random.Random(f"{train.TRAIN_SEED}:{salt}")
    items = list(items)
    rng.shuffle(items)
    return items[:count]


def create_fixed_eval_set() -> list[dict]:
    self_items = []
    for item in train.read_json_list(train.SELF_PLAY_BUFFER):
        sample = _valid_self_play_sample(item)
        if sample is not None:
            self_items.append(sample)

    stockfish_items = []
    for item in train.normalize_labels(train.read_json_list(train.STOCKFISH_REPLAY_BUFFER)):
        sample = _valid_stockfish_sample(item)
        if sample is not None:
            stockfish_items.append(sample)

    fixed_items = _pick_deterministic(self_items, FIXED_EVAL_SELF, "self")
    fixed_items += _pick_deterministic(stockfish_items, FIXED_EVAL_STOCKFISH, "stockfish")
    if not fixed_items:
        raise ValueError("could not create fixed evaluation set: no valid samples")

    fixed_items = _pick_deterministic(fixed_items, len(fixed_items), "mixed")
    train.write_json(FIXED_EVAL_SET, fixed_items)
    return fixed_items


def load_fixed_eval_set() -> list[dict]:
    items = train.read_json_list(FIXED_EVAL_SET)
    if items:
        return items
    return create_fixed_eval_set()


def fixed_eval_arrays():
    samples = load_fixed_eval_set()
    X = []
    policy_y = []
    value_y = []
    policy_weights = []
    value_weights = []
    self_count = 0
    stockfish_count = 0

    zero_policy = np.zeros((train.POLICY_SIZE,), dtype=np.float32)
    for sample in samples:
        source = sample.get("source")
        fen = sample.get("fen")
        if not fen:
            continue
        X.append(train.board_to_features(fen))
        if source == "self_play":
            policy = train.dense_policy_from_sparse(sample.get("policy"))
            try:
                value = float(sample.get("z"))
            except (TypeError, ValueError):
                continue
            policy_y.append(policy)
            value_y.append(np.clip(value, -1.0, 1.0))
            policy_weights.append(1.0)
            value_weights.append(1.0)
            self_count += 1
        elif source == "stockfish":
            try:
                cp = float(sample.get("cp", 0.0))
            except (TypeError, ValueError):
                continue
            policy_y.append(zero_policy.copy())
            value_y.append(train.cp_to_value(cp))
            policy_weights.append(0.0)
            value_weights.append(1.0)
            stockfish_count += 1
        else:
            continue

    if not X:
        raise ValueError("fixed evaluation set has no usable samples")

    print(
        f"[train] fixed eval set {len(X)} positions "
        f"(self-play {self_count}, Stockfish {stockfish_count})"
    )
    return (
        np.stack(X).astype(np.float32),
        np.stack(policy_y).astype(np.float32),
        np.array(value_y, dtype=np.float32).reshape(-1, 1),
        np.array(policy_weights, dtype=np.float32),
        np.array(value_weights, dtype=np.float32),
    )


def main():
    X, policy_y, value_y, policy_weights, value_weights, self_play_count, fresh_count, stockfish_count = train.load_dataset()
    X = train.ensure_4d_board(X)

    (Xtr, Xva), (Ptr, Pva), (Vtr, Vva), (PWtr, PWva), (VWtr, VWva) = train.split_arrays(
        X,
        policy_y,
        value_y,
        policy_weights,
        value_weights,
    )
    Xev, Pev, Vev, PWev, VWev = fixed_eval_arrays()

    print(f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, self-play {self_play_count}, Stockfish replay {stockfish_count}")

    baseline_model = train.load_saved_dual_head_model(train.CONTINUE_LR, quiet=True)
    model, resumed = train.load_or_build_model(X.shape[1:])
    epochs = train.CONTINUE_EPOCHS if resumed else train.COLD_START_EPOCHS

    moving_baseline_eval = train.evaluate_model(baseline_model, Xva, Pva, Vva, PWva, VWva, "previous moving")
    fixed_baseline_eval = train.evaluate_model(baseline_model, Xev, Pev, Vev, PWev, VWev, "previous fixed")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        Xtr,
        [Ptr, Vtr],
        validation_data=(Xva, [Pva, Vva], [PWva, VWva]),
        sample_weight=[PWtr, VWtr],
        epochs=epochs,
        batch_size=256,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    moving_candidate_eval = train.evaluate_model(model, Xva, Pva, Vva, PWva, VWva, "candidate moving")
    fixed_candidate_eval = train.evaluate_model(model, Xev, Pev, Vev, PWev, VWev, "candidate fixed")
    accepted, gate_reason = train.should_accept_candidate(fixed_candidate_eval, fixed_baseline_eval, resumed)
    print(f"[train] candidate gate: accepted={accepted} reason=fixed_eval_{gate_reason}")

    if accepted:
        model.save(train.CHECKPOINT_MODEL)
        print(f"[train] saved accepted brain to {train.CHECKPOINT_MODEL}")

        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, str(train.OUT_DIR))

        model_json = train.OUT_DIR / "model.json"
        train.patch_tfjs_model_json(model_json)
        train.smoke_check_tfjs_json(model_json)
        print(f"[train] saved accepted TFJS model to {model_json}")
    else:
        print("[train] rejected candidate; keeping previous checkpoint and browser model")

    train.append_training_history(
        history,
        resumed,
        self_play_count,
        fresh_count,
        stockfish_count,
        epochs,
        accepted,
        f"fixed_eval_{gate_reason}",
        fixed_baseline_eval,
        fixed_candidate_eval,
    )

    # Log moving-split metrics for debugging without using them for the accept gate.
    if moving_baseline_eval and moving_candidate_eval:
        baseline_loss = moving_baseline_eval.get("loss")
        candidate_loss = moving_candidate_eval.get("loss")
        print(f"[train] moving validation was diagnostic only: previous={baseline_loss:.6f}, candidate={candidate_loss:.6f}")


if __name__ == "__main__":
    main()
