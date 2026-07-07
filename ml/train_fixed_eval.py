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
_EPS = 1e-7


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

        if source == "self_play":
            policy = train.dense_policy_from_sparse(sample.get("policy"))
            if float(np.sum(policy)) <= 0:
                continue
            try:
                value = float(sample.get("z"))
            except (TypeError, ValueError):
                continue
            X.append(train.board_to_features(fen))
            policy_y.append(policy.astype(np.float32))
            value_y.append(np.clip(value, -1.0, 1.0))
            policy_weights.append(1.0)
            value_weights.append(1.0)
            self_count += 1
        elif source == "stockfish":
            try:
                cp = float(sample.get("cp", 0.0))
            except (TypeError, ValueError):
                continue
            X.append(train.board_to_features(fen))
            policy_y.append(zero_policy.copy())
            value_y.append(train.cp_to_value(cp))
            policy_weights.append(0.0)
            value_weights.append(1.0)
            stockfish_count += 1

    if not X:
        raise ValueError("fixed evaluation set has no usable samples")

    X_arr = train.ensure_4d_board(np.stack(X).astype(np.float32))
    P_arr = np.stack(policy_y).astype(np.float32)
    V_arr = np.array(value_y, dtype=np.float32).reshape(-1, 1)
    PW_arr = np.array(policy_weights, dtype=np.float32)
    VW_arr = np.array(value_weights, dtype=np.float32)

    if not (len(X_arr) == len(P_arr) == len(V_arr) == len(PW_arr) == len(VW_arr)):
        raise ValueError("fixed evaluation arrays have inconsistent lengths")

    print(
        f"[train] fixed eval set {len(X_arr)} positions "
        f"(self-play {self_count}, Stockfish {stockfish_count})"
    )
    return X_arr, P_arr, V_arr, PW_arr, VW_arr


def _softmax_cross_entropy(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True) + _EPS)
    return -np.sum(labels * log_probs, axis=1)


def evaluate_fixed_model(model: tf.keras.Model | None, X, P, V, PW, VW, label: str) -> dict | None:
    if model is None:
        return None
    try:
        predictions = model.predict(X, batch_size=256, verbose=0)
    except Exception as exc:
        print(f"[train] could not predict with {label} model: {exc}")
        return None

    if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
        print(f"[train] could not evaluate {label} model: expected two outputs")
        return None

    policy_logits = np.asarray(predictions[0], dtype=np.float32)
    value_pred = np.asarray(predictions[1], dtype=np.float32).reshape(-1, 1)

    policy_den = max(float(np.sum(PW)), _EPS)
    value_den = max(float(np.sum(VW)), _EPS)
    policy_loss = float(np.sum(_softmax_cross_entropy(P, policy_logits) * PW) / policy_den)
    value_errors = np.square(V - value_pred).reshape(-1)
    value_abs_errors = np.abs(V - value_pred).reshape(-1)
    value_loss = float(np.sum(value_errors * VW) / value_den)
    value_mae = float(np.sum(value_abs_errors * VW) / value_den)
    total_loss = policy_loss + value_loss

    metrics = {
        "loss": total_loss,
        "policy_logits_loss": policy_loss,
        "value_loss": value_loss,
        "value_mae": value_mae,
    }
    print(
        f"[train] {label} validation loss {total_loss:.6f} "
        f"(policy {policy_loss:.6f}, value {value_loss:.6f}, value_mae {value_mae:.6f})"
    )
    return metrics


def accept_from_fixed_eval(candidate_eval: dict | None, baseline_eval: dict | None, resumed: bool) -> tuple[bool, str]:
    if candidate_eval is None:
        return False, "fixed_eval_candidate_failed"
    if resumed and baseline_eval is None:
        return False, "fixed_eval_previous_failed"
    accepted, reason = train.should_accept_candidate(candidate_eval, baseline_eval, resumed)
    return accepted, f"fixed_eval_{reason}"


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
    fixed_baseline_eval = evaluate_fixed_model(baseline_model, Xev, Pev, Vev, PWev, VWev, "previous fixed")

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
    fixed_candidate_eval = evaluate_fixed_model(model, Xev, Pev, Vev, PWev, VWev, "candidate fixed")
    accepted, gate_reason = accept_from_fixed_eval(fixed_candidate_eval, fixed_baseline_eval, resumed)
    print(f"[train] candidate gate: accepted={accepted} reason={gate_reason}")

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
        gate_reason,
        fixed_baseline_eval,
        fixed_candidate_eval,
    )

    if moving_baseline_eval and moving_candidate_eval:
        baseline_loss = moving_baseline_eval.get("loss")
        candidate_loss = moving_candidate_eval.get("loss")
        print(f"[train] moving validation was diagnostic only: previous={baseline_loss:.6f}, candidate={candidate_loss:.6f}")


if __name__ == "__main__":
    main()
