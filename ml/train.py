# ml/train.py
"""
train the chess model and keep a saved brain for the next run
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import datetime as dt
import json
import pathlib
import random

import numpy as np
import tensorflow as tf
from features import board_to_features
from policy_map import POLICY_SIZE

LABELS = pathlib.Path("ml/data/labels.json")
STOCKFISH_REPLAY_BUFFER = pathlib.Path("ml/data/replay_buffer.json")
SELF_PLAY_BUFFER = pathlib.Path("ml/data/self_play_buffer.json")
TRAINING_HISTORY = pathlib.Path("ml/training_history.json")
CHECKPOINT_DIR = pathlib.Path("ml/checkpoints")
CHECKPOINT_MODEL = CHECKPOINT_DIR / "chess_eval.keras"
OUT_DIR = pathlib.Path("public/nn")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BOARD_H, BOARD_W, PLANES = 8, 8, 13
FLAT_SIZE = BOARD_H * BOARD_W * PLANES
MAX_REPLAY_ITEMS = int(os.environ.get("MAX_REPLAY_ITEMS", "50000"))
MAX_SELF_PLAY_TRAIN = int(os.environ.get("AZ_MAX_SELF_PLAY_TRAIN", "4000"))
MAX_STOCKFISH_TRAIN = int(os.environ.get("AZ_MAX_STOCKFISH_TRAIN", "8000"))
COLD_START_EPOCHS = int(os.environ.get("COLD_START_EPOCHS", "6"))
CONTINUE_EPOCHS = int(os.environ.get("CONTINUE_EPOCHS", "3"))
COLD_START_LR = float(os.environ.get("COLD_START_LR", "1e-3"))
CONTINUE_LR = float(os.environ.get("CONTINUE_LR", "2e-4"))
MIN_VALIDATION_IMPROVEMENT = float(os.environ.get("AZ_MIN_VALIDATION_IMPROVEMENT", "0.0"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))

random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
tf.keras.utils.set_random_seed(TRAIN_SEED)


def read_json_list(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def normalize_labels(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        fen = item.get("fen")
        if not fen:
            continue
        try:
            cp = float(item.get("cp", 0.0))
        except (TypeError, ValueError):
            continue
        normalized.append({"fen": fen, "cp": cp})
    return normalized


def merge_stockfish_replay_buffer(new_items: list[dict]) -> list[dict]:
    existing_items = normalize_labels(read_json_list(STOCKFISH_REPLAY_BUFFER))
    new_items = normalize_labels(new_items)

    by_fen = {item["fen"]: item for item in existing_items}
    for item in new_items:
        by_fen[item["fen"]] = item

    new_by_fen = {item["fen"]: item for item in new_items}
    if len(new_by_fen) >= MAX_REPLAY_ITEMS:
        merged = list(new_by_fen.values())
        random.shuffle(merged)
        merged = merged[:MAX_REPLAY_ITEMS]
    else:
        old_items = [item for fen, item in by_fen.items() if fen not in new_by_fen]
        random.shuffle(old_items)
        room_for_old = MAX_REPLAY_ITEMS - len(new_by_fen)
        merged = list(new_by_fen.values()) + old_items[:room_for_old]

    random.shuffle(merged)
    write_json(STOCKFISH_REPLAY_BUFFER, merged)
    print(f"[train] Stockfish replay buffer {len(merged)} positions, new {len(new_items)}")
    return merged


def cp_to_value(cp: float) -> float:
    return float(np.tanh(np.clip(cp, -2000.0, 2000.0) / 600.0))


def dense_policy_from_sparse(policy_items) -> np.ndarray:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for item in policy_items or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        idx, prob = item
        try:
            idx = int(idx)
            prob = float(prob)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < POLICY_SIZE and prob > 0:
            policy[idx] += prob
    total = float(np.sum(policy))
    if total > 0:
        policy /= total
    return policy


def load_self_play_samples():
    items = read_json_list(SELF_PLAY_BUFFER)[-MAX_SELF_PLAY_TRAIN:]
    X, policies, values = [], [], []
    for item in items:
        fen = item.get("fen")
        if not fen:
            continue
        policy = dense_policy_from_sparse(item.get("policy"))
        if float(np.sum(policy)) <= 0:
            continue
        try:
            z = float(item.get("z"))
        except (TypeError, ValueError):
            continue
        X.append(board_to_features(fen))
        policies.append(policy)
        values.append(np.clip(z, -1.0, 1.0))
    return X, policies, values


def load_stockfish_samples():
    fresh_items = normalize_labels(read_json_list(LABELS))
    items = merge_stockfish_replay_buffer(fresh_items)[-MAX_STOCKFISH_TRAIN:]
    X, values = [], []
    for item in items:
        X.append(board_to_features(item["fen"]))
        values.append(cp_to_value(float(item["cp"])))
    return X, values, len(fresh_items), len(items)


def load_dataset():
    self_X, self_policy, self_value = load_self_play_samples()
    stock_X, stock_value, fresh_count, stockfish_count = load_stockfish_samples()

    X = []
    policy_y = []
    value_y = []
    policy_weights = []
    value_weights = []

    for x, policy, value in zip(self_X, self_policy, self_value):
        X.append(x)
        policy_y.append(policy)
        value_y.append(value)
        policy_weights.append(1.0)
        value_weights.append(1.0)

    zero_policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for x, value in zip(stock_X, stock_value):
        X.append(x)
        policy_y.append(zero_policy.copy())
        value_y.append(value)
        policy_weights.append(0.0)
        value_weights.append(1.0)

    if not X:
        raise ValueError("no training samples found")

    combined = list(zip(X, policy_y, value_y, policy_weights, value_weights))
    random.shuffle(combined)
    X, policy_y, value_y, policy_weights, value_weights = zip(*combined)

    return (
        np.stack(X).astype(np.float32),
        np.stack(policy_y).astype(np.float32),
        np.array(value_y, dtype=np.float32).reshape(-1, 1),
        np.array(policy_weights, dtype=np.float32),
        np.array(value_weights, dtype=np.float32),
        len(self_X),
        fresh_count,
        stockfish_count,
    )


def ensure_4d_board(X: np.ndarray) -> np.ndarray:
    if X.ndim == 4 and X.shape[1:] == (BOARD_H, BOARD_W, PLANES):
        return X
    if X.ndim == 2 and X.shape[1] == FLAT_SIZE:
        return X.reshape(-1, BOARD_H, BOARD_W, PLANES)
    raise ValueError(
        f"Expected X as (N, {BOARD_H}, {BOARD_W}, {PLANES}) or (N, {FLAT_SIZE}), "
        f"but got {X.shape}. Check features.board_to_features"
    )


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=[
            tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            tf.keras.losses.MeanSquaredError(),
        ],
        loss_weights=[1.0, 1.0],
        metrics=[[], [tf.keras.metrics.MeanAbsoluteError(name="mae")]],
    )
    return model


def build_model(input_shape=(BOARD_H, BOARD_W, PLANES), learning_rate=COLD_START_LR):
    inp = tf.keras.Input(shape=input_shape, name="board")
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    policy_logits = tf.keras.layers.Dense(POLICY_SIZE, activation="linear", name="policy_logits")(x)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(x)
    model = tf.keras.Model(inp, [policy_logits, value])
    return compile_model(model, learning_rate)


def is_dual_head_model(model: tf.keras.Model) -> bool:
    if len(model.outputs) != 2:
        return False
    output_names = set(getattr(model, "output_names", []))
    return not output_names or {"policy_logits", "value"}.issubset(output_names)


def load_saved_dual_head_model(learning_rate: float, *, quiet: bool = False) -> tf.keras.Model | None:
    if not CHECKPOINT_MODEL.exists():
        return None
    try:
        model = tf.keras.models.load_model(CHECKPOINT_MODEL, compile=False)
    except Exception as exc:
        if not quiet:
            print(f"[train] could not load saved brain, starting fresh: {exc}")
        return None

    if not is_dual_head_model(model):
        if not quiet:
            print("[train] saved brain is value-only, starting a new dual-head brain")
        return None

    return compile_model(model, learning_rate)


def load_or_build_model(input_shape):
    model = load_saved_dual_head_model(CONTINUE_LR)
    if model is not None:
        print(f"[train] loading saved dual-head brain from {CHECKPOINT_MODEL}")
        return model, True

    print("[train] no compatible saved brain found, starting from scratch")
    return build_model(input_shape, COLD_START_LR), False


def evaluate_model(model: tf.keras.Model | None, X, P, V, PW, VW, label: str) -> dict | None:
    if model is None:
        return None
    try:
        metrics = model.evaluate(
            X,
            [P, V],
            sample_weight=[PW, VW],
            batch_size=256,
            verbose=0,
            return_dict=True,
        )
    except Exception as exc:
        print(f"[train] could not evaluate {label} model: {exc}")
        return None

    metrics = {key: float(value) for key, value in metrics.items()}
    loss = metrics.get("loss")
    if loss is not None:
        print(f"[train] {label} validation loss {loss:.6f}")
    else:
        print(f"[train] {label} validation metrics {metrics}")
    return metrics


def should_accept_candidate(candidate_eval: dict | None, baseline_eval: dict | None, resumed: bool) -> tuple[bool, str]:
    if not resumed or baseline_eval is None:
        return True, "no_previous_compatible_checkpoint"
    if candidate_eval is None:
        return False, "candidate_validation_failed"

    baseline_loss = baseline_eval.get("loss")
    candidate_loss = candidate_eval.get("loss")
    if baseline_loss is None or not np.isfinite(baseline_loss):
        return True, "previous_validation_loss_unavailable"
    if candidate_loss is None or not np.isfinite(candidate_loss):
        return False, "candidate_validation_loss_unavailable"

    required_loss = baseline_loss - MIN_VALIDATION_IMPROVEMENT
    if candidate_loss <= required_loss:
        return True, "candidate_validation_loss_improved"
    return False, "candidate_validation_loss_did_not_improve"


def patch_tfjs_model_json(path: pathlib.Path) -> None:
    j = json.loads(path.read_text(encoding="utf-8"))

    cfg = (
        j.get("modelTopology", {})
         .get("model_config", {})
         .get("config", {})
    )
    layers = cfg.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("model.json missing modelTopology.model_config.config.layers")

    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            c = layer.get("config", {})
            if "batch_shape" in c and "batchInputShape" not in c:
                c["batchInputShape"] = c.pop("batch_shape")

    def get_history(arg):
        if isinstance(arg, dict):
            if isinstance(arg.get("keras_history"), list):
                return arg["keras_history"]
            c2 = arg.get("config")
            if isinstance(c2, dict) and isinstance(c2.get("keras_history"), list):
                return c2["keras_history"]
        return None

    for layer in layers:
        inbound = layer.get("inbound_nodes")
        if not isinstance(inbound, list):
            continue

        if len(inbound) > 0 and all(isinstance(x, list) for x in inbound):
            continue

        new_inbound = []
        for node in inbound:
            if isinstance(node, dict) and isinstance(node.get("args"), list):
                conns = []
                for a in node["args"]:
                    h = get_history(a)
                    if h and len(h) >= 3:
                        lname, nidx, tidx = h[:3]
                        conns.append([lname, nidx, tidx, {}])
                new_inbound.append(conns)
            else:
                new_inbound.append([])
        layer["inbound_nodes"] = new_inbound

    il = cfg.get("input_layers")
    if isinstance(il, list) and len(il) == 3 and isinstance(il[0], str):
        cfg["input_layers"] = [il]
    ol = cfg.get("output_layers")
    if isinstance(ol, list) and len(ol) == 3 and isinstance(ol[0], str):
        cfg["output_layers"] = [ol]

    path.write_text(json.dumps(j), encoding="utf-8")


def smoke_check_tfjs_json(path: pathlib.Path) -> None:
    j = json.loads(path.read_text(encoding="utf-8"))
    cfg = j["modelTopology"]["model_config"]["config"]

    il = cfg["input_layers"]
    ol = cfg["output_layers"]
    assert isinstance(il, list) and len(il) > 0 and isinstance(il[0], list), "input_layers not nested"
    assert isinstance(ol, list) and len(ol) > 0 and isinstance(ol[0], list), "output_layers not nested"

    for layer in cfg["layers"]:
        inbound = layer.get("inbound_nodes", [])
        if not isinstance(inbound, list):
            raise AssertionError("inbound_nodes missing or invalid")
        for node in inbound:
            if not isinstance(node, list):
                raise AssertionError("inbound_nodes contains non-list node")

    for layer in cfg["layers"]:
        if layer.get("class_name") == "InputLayer":
            c = layer.get("config", {})
            assert "batchInputShape" in c, "InputLayer missing batchInputShape"


def add_eval_metrics(record: dict, prefix: str, metrics: dict | None) -> None:
    if not metrics:
        return
    for key, value in metrics.items():
        try:
            record[f"{prefix}_{key}"] = float(value)
        except (TypeError, ValueError):
            continue


def append_training_history(
    history,
    resumed,
    self_play_count,
    fresh_count,
    stockfish_count,
    epochs,
    accepted,
    gate_reason,
    baseline_eval,
    candidate_eval,
):
    records = read_json_list(TRAINING_HISTORY)
    record = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "resumed_from_checkpoint": resumed,
        "self_play_positions": self_play_count,
        "fresh_stockfish_positions": fresh_count,
        "stockfish_replay_positions": stockfish_count,
        "epochs": epochs,
        "candidate_accepted": accepted,
        "gate_reason": gate_reason,
        "min_validation_improvement": MIN_VALIDATION_IMPROVEMENT,
    }

    for key, values in history.history.items():
        if values:
            record[f"final_{key}"] = float(values[-1])

    add_eval_metrics(record, "baseline_validation", baseline_eval)
    add_eval_metrics(record, "candidate_validation", candidate_eval)

    records.append(record)
    write_json(TRAINING_HISTORY, records[-365:])


def split_arrays(*arrays, train_ratio=0.9):
    n = len(arrays[0])
    split = max(1, int(train_ratio * n))
    if split >= n:
        split = n - 1 if n > 1 else n
    return [(arr[:split], arr[split:]) for arr in arrays]


def main():
    X, policy_y, value_y, policy_weights, value_weights, self_play_count, fresh_count, stockfish_count = load_dataset()
    X = ensure_4d_board(X)

    (Xtr, Xva), (Ptr, Pva), (Vtr, Vva), (PWtr, PWva), (VWtr, VWva) = split_arrays(
        X,
        policy_y,
        value_y,
        policy_weights,
        value_weights,
    )

    print(f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, self-play {self_play_count}, Stockfish replay {stockfish_count}")

    baseline_model = load_saved_dual_head_model(CONTINUE_LR, quiet=True)
    model, resumed = load_or_build_model(X.shape[1:])
    epochs = CONTINUE_EPOCHS if resumed else COLD_START_EPOCHS

    baseline_eval = evaluate_model(baseline_model, Xva, Pva, Vva, PWva, VWva, "previous")

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

    candidate_eval = evaluate_model(model, Xva, Pva, Vva, PWva, VWva, "candidate")
    accepted, gate_reason = should_accept_candidate(candidate_eval, baseline_eval, resumed)
    print(f"[train] candidate gate: accepted={accepted} reason={gate_reason}")

    if accepted:
        model.save(CHECKPOINT_MODEL)
        print(f"[train] saved accepted brain to {CHECKPOINT_MODEL}")

        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, str(OUT_DIR))

        model_json = OUT_DIR / "model.json"
        patch_tfjs_model_json(model_json)
        smoke_check_tfjs_json(model_json)
        print(f"[train] saved accepted TFJS model to {model_json}")
    else:
        print("[train] rejected candidate; keeping previous checkpoint and browser model")

    append_training_history(
        history,
        resumed,
        self_play_count,
        fresh_count,
        stockfish_count,
        epochs,
        accepted,
        gate_reason,
        baseline_eval,
        candidate_eval,
    )


if __name__ == "__main__":
    main()
