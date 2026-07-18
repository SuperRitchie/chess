# ml/train.py
"""Train the chess policy/value model and preserve a compatible checkpoint."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import datetime as dt
import json
import pathlib
import random

import chess
import numpy as np
import tensorflow as tf

from features import PLANES, board_to_features
from fen_utils import canonical_fen
from policy_map import (
    LEGACY_POLICY_SIZE,
    POLICY_CHANNELS,
    POLICY_SIZE,
    POLICY_VERSION,
    normalize_policy_index,
)

LABELS = pathlib.Path("ml/data/labels.json")
STOCKFISH_REPLAY_BUFFER = pathlib.Path("ml/data/replay_buffer.json")
SELF_PLAY_BUFFER = pathlib.Path("ml/data/self_play_buffer.json")
TRAINING_HISTORY = pathlib.Path("ml/training_history.json")
CHECKPOINT_DIR = pathlib.Path("ml/checkpoints")
CHECKPOINT_MODEL = CHECKPOINT_DIR / "chess_eval.keras"
OUT_DIR = pathlib.Path("public/nn")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BOARD_H, BOARD_W = 8, 8
FLAT_SIZE = BOARD_H * BOARD_W * PLANES
MAX_REPLAY_ITEMS = int(os.environ.get("MAX_REPLAY_ITEMS", "50000"))
MAX_SELF_PLAY_TRAIN = int(os.environ.get("AZ_MAX_SELF_PLAY_TRAIN", "12000"))
MAX_STOCKFISH_TRAIN = int(os.environ.get("AZ_MAX_STOCKFISH_TRAIN", "12000"))
COLD_START_EPOCHS = int(os.environ.get("COLD_START_EPOCHS", "6"))
CONTINUE_EPOCHS = int(os.environ.get("CONTINUE_EPOCHS", "3"))
COLD_START_LR = float(os.environ.get("COLD_START_LR", "1e-3"))
CONTINUE_LR = float(os.environ.get("CONTINUE_LR", "2e-4"))
MIN_VALIDATION_IMPROVEMENT = float(os.environ.get("AZ_MIN_VALIDATION_IMPROVEMENT", "0.0001"))
STOCKFISH_VALUE_WEIGHT = float(os.environ.get("AZ_STOCKFISH_VALUE_WEIGHT", "1.0"))
STOCKFISH_POLICY_WEIGHT = float(os.environ.get("AZ_STOCKFISH_POLICY_WEIGHT", "1.0"))
MERGE_FRESH_STOCKFISH_LABELS = os.environ.get("AZ_MERGE_FRESH_STOCKFISH_LABELS", "1") != "0"
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


def normalize_sparse_policy(policy_items, fen: str, policy_version: int) -> list[list[float]]:
    board = chess.Board(fen)
    by_index = {}
    for item in policy_items or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            index = normalize_policy_index(int(item[0]), board, policy_version)
            probability = float(item[1])
        except (TypeError, ValueError):
            continue
        if probability > 0 and np.isfinite(probability):
            by_index[index] = by_index.get(index, 0.0) + probability

    total = sum(by_index.values())
    if total <= 0:
        return []
    return [[index, probability / total] for index, probability in sorted(by_index.items())]


def normalize_labels(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        fen = item.get("fen")
        if not fen:
            continue
        try:
            fen = canonical_fen(fen)
            cp = float(item.get("cp", 0.0))
            policy_version = int(item.get("policy_version", 1))
        except (TypeError, ValueError):
            continue
        policy = normalize_sparse_policy(item.get("policy"), fen, policy_version)
        normalized.append(
            {
                "fen": fen,
                "cp": cp,
                "policy_version": POLICY_VERSION if policy else policy_version,
                "policy": policy,
            }
        )
    return normalized


def merge_stockfish_replay_buffer(new_items: list[dict]) -> tuple[list[dict], int]:
    existing_items = normalize_labels(read_json_list(STOCKFISH_REPLAY_BUFFER))
    new_items = normalize_labels(new_items)

    existing_by_fen = {item["fen"]: item for item in existing_items}
    new_by_fen = {item["fen"]: item for item in new_items}
    novel_count = sum(fen not in existing_by_fen for fen in new_by_fen)
    by_fen = dict(existing_by_fen)
    for fen, item in list(new_by_fen.items()):
        existing = by_fen.get(item["fen"])
        if existing and existing.get("policy") and not item.get("policy"):
            item = {**item, "policy_version": existing["policy_version"], "policy": existing["policy"]}
        new_by_fen[fen] = item
        by_fen[item["fen"]] = item

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
    policy_count = sum(bool(item.get("policy")) for item in merged)
    print(
        f"[train] Stockfish replay buffer {len(merged)} positions, "
        f"incoming {len(new_by_fen)}, truly new {novel_count}, policy targets {policy_count}"
    )
    return merged, novel_count


def cp_to_value(cp: float) -> float:
    return float(np.tanh(np.clip(cp, -2000.0, 2000.0) / 600.0))


def dense_policy_from_sparse(policy_items, fen: str | None = None, policy_version: int = 1) -> np.ndarray:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    board = chess.Board(fen) if fen else None
    for item in policy_items or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        raw_index, probability = item
        try:
            index = normalize_policy_index(int(raw_index), board, int(policy_version))
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if probability > 0:
            policy[index] += probability
    total = float(np.sum(policy))
    if total > 0:
        policy /= total
    return policy


def load_self_play_samples(excluded_fens: set[str] | None = None):
    excluded_fens = excluded_fens or set()
    items = read_json_list(SELF_PLAY_BUFFER)[-MAX_SELF_PLAY_TRAIN:]
    X, policies, values = [], [], []
    for item in items:
        fen = item.get("fen")
        if not fen or fen in excluded_fens:
            continue
        policy = dense_policy_from_sparse(
            item.get("policy"),
            fen=fen,
            policy_version=int(item.get("policy_version", 1)),
        )
        if float(np.sum(policy)) <= 0:
            continue
        try:
            outcome = float(item.get("z"))
        except (TypeError, ValueError):
            continue
        X.append(board_to_features(fen))
        policies.append(policy)
        values.append(np.clip(outcome, -1.0, 1.0))
    return X, policies, values


def load_stockfish_samples(excluded_fens: set[str] | None = None):
    excluded_fens = excluded_fens or set()
    fresh_items = normalize_labels(read_json_list(LABELS)) if MERGE_FRESH_STOCKFISH_LABELS else []
    all_items, novel_count = merge_stockfish_replay_buffer(fresh_items)
    items = [item for item in all_items if item["fen"] not in excluded_fens][-MAX_STOCKFISH_TRAIN:]
    X, policies, values, policy_weights = [], [], [], []
    for item in items:
        X.append(board_to_features(item["fen"]))
        policy = dense_policy_from_sparse(
            item.get("policy"),
            fen=item["fen"],
            policy_version=int(item.get("policy_version", POLICY_VERSION)),
        )
        policies.append(policy)
        policy_weights.append(STOCKFISH_POLICY_WEIGHT if float(np.sum(policy)) > 0 else 0.0)
        values.append(cp_to_value(float(item["cp"])))
    print(f"[train] using {sum(weight > 0 for weight in policy_weights)} Stockfish policy targets")
    return X, policies, values, policy_weights, novel_count, len(items)


def load_dataset(excluded_fens: set[str] | None = None):
    self_X, self_policy, self_value = load_self_play_samples(excluded_fens)
    stock_X, stock_policy, stock_value, stock_policy_weights, fresh_count, stockfish_count = load_stockfish_samples(
        excluded_fens
    )

    X = []
    policy_y = []
    value_y = []
    policy_weights = []
    value_weights = []

    for features, policy, value in zip(self_X, self_policy, self_value):
        X.append(features)
        policy_y.append(policy)
        value_y.append(value)
        policy_weights.append(1.0)
        value_weights.append(1.0)

    for features, policy, value, policy_weight in zip(
        stock_X,
        stock_policy,
        stock_value,
        stock_policy_weights,
    ):
        X.append(features)
        policy_y.append(policy)
        value_y.append(value)
        policy_weights.append(policy_weight)
        value_weights.append(STOCKFISH_VALUE_WEIGHT)

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
    inputs = tf.keras.Input(shape=input_shape, name="board")
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", name="trunk_conv_1")(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", name="trunk_conv_2")(x)
    x = tf.keras.layers.Flatten(name="trunk_flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="trunk_dense")(x)
    policy_logits = tf.keras.layers.Dense(POLICY_SIZE, activation="linear", name="policy_logits")(x)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(x)
    model = tf.keras.Model(inputs, [policy_logits, value])
    return compile_model(model, learning_rate)


def is_dual_head_model(model: tf.keras.Model) -> bool:
    try:
        input_shape = tuple(model.input_shape[1:])
        policy_size = int(model.outputs[0].shape[-1])
        value_size = int(model.outputs[1].shape[-1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return False
    output_names = set(getattr(model, "output_names", []))
    names_ok = not output_names or {"policy_logits", "value"}.issubset(output_names)
    return (
        len(model.outputs) == 2
        and names_ok
        and input_shape == (BOARD_H, BOARD_W, PLANES)
        and policy_size == POLICY_SIZE
        and value_size == 1
    )


def _find_layer(model, *names):
    for name in names:
        try:
            return model.get_layer(name)
        except ValueError:
            continue
    return None


def migrate_legacy_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model | None:
    """Transfer useful v1 weights into the expanded feature/action model."""
    try:
        old_input_shape = tuple(model.input_shape[1:])
        old_policy_size = int(model.outputs[0].shape[-1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if old_input_shape != (8, 8, 13) or old_policy_size != LEGACY_POLICY_SIZE:
        return None

    migrated = build_model((BOARD_H, BOARD_W, PLANES), learning_rate)
    old_conv1 = _find_layer(model, "trunk_conv_1", "conv2d")
    old_conv2 = _find_layer(model, "trunk_conv_2", "conv2d_1")
    old_dense = _find_layer(model, "trunk_dense", "dense")
    old_policy = _find_layer(model, "policy_logits")
    old_value = _find_layer(model, "value")
    if not all((old_conv1, old_conv2, old_dense, old_policy, old_value)):
        return None

    old_kernel, old_bias = old_conv1.get_weights()
    new_kernel, _ = migrated.get_layer("trunk_conv_1").get_weights()
    new_kernel[:, :, : old_kernel.shape[2], :] = old_kernel
    migrated.get_layer("trunk_conv_1").set_weights([new_kernel, old_bias])
    migrated.get_layer("trunk_conv_2").set_weights(old_conv2.get_weights())
    migrated.get_layer("trunk_dense").set_weights(old_dense.get_weights())
    migrated.get_layer("value").set_weights(old_value.get_weights())

    old_policy_kernel, old_policy_bias = old_policy.get_weights()
    new_policy_kernel = np.repeat(old_policy_kernel, POLICY_CHANNELS, axis=1)
    new_policy_bias = np.repeat(old_policy_bias, POLICY_CHANNELS)
    migrated.get_layer("policy_logits").set_weights([new_policy_kernel, new_policy_bias])
    print("[train] migrated legacy 13-plane/4096-action checkpoint to the v2 representation")
    return migrated


def load_checkpoint_raw(*, compile_saved: bool, quiet: bool = False):
    if not CHECKPOINT_MODEL.exists():
        return None
    try:
        return tf.keras.models.load_model(CHECKPOINT_MODEL, compile=compile_saved)
    except Exception as exc:
        if not quiet:
            print(f"[train] could not load saved brain: {exc}")
        return None


def load_saved_dual_head_model(
    learning_rate: float,
    *,
    quiet: bool = False,
    preserve_optimizer: bool = False,
) -> tf.keras.Model | None:
    model = load_checkpoint_raw(compile_saved=preserve_optimizer, quiet=quiet)
    if model is None or not is_dual_head_model(model):
        return None

    if preserve_optimizer and getattr(model, "optimizer", None) is not None:
        try:
            model.optimizer.learning_rate.assign(learning_rate)
        except (AttributeError, TypeError, ValueError):
            tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        return model
    return compile_model(model, learning_rate)


def load_or_build_model(input_shape):
    model = load_saved_dual_head_model(CONTINUE_LR, preserve_optimizer=True)
    if model is not None:
        print(f"[train] loading saved dual-head brain and optimizer from {CHECKPOINT_MODEL}")
        return model, True

    legacy = load_checkpoint_raw(compile_saved=False, quiet=True)
    if legacy is not None:
        migrated = migrate_legacy_model(legacy, CONTINUE_LR)
        if migrated is not None:
            return migrated, False

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
    if candidate_eval is None:
        return False, "candidate_validation_failed"
    if resumed and baseline_eval is None:
        return False, "previous_validation_failed"
    if not resumed or baseline_eval is None:
        return True, "no_previous_compatible_checkpoint"

    baseline_loss = baseline_eval.get("loss")
    candidate_loss = candidate_eval.get("loss")
    if baseline_loss is None or not np.isfinite(baseline_loss):
        return False, "previous_validation_loss_unavailable"
    if candidate_loss is None or not np.isfinite(candidate_loss):
        return False, "candidate_validation_loss_unavailable"

    required_loss = baseline_loss - MIN_VALIDATION_IMPROVEMENT
    if candidate_loss <= required_loss:
        return True, "candidate_validation_loss_improved"
    return False, "candidate_validation_loss_did_not_improve"


def patch_tfjs_model_json(path: pathlib.Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("modelTopology", {}).get("model_config", {}).get("config", {})
    layers = config.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("model.json missing modelTopology.model_config.config.layers")

    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            layer_config = layer.get("config", {})
            if "batch_shape" in layer_config and "batchInputShape" not in layer_config:
                layer_config["batchInputShape"] = layer_config.pop("batch_shape")

    def get_history(argument):
        if isinstance(argument, dict):
            if isinstance(argument.get("keras_history"), list):
                return argument["keras_history"]
            nested = argument.get("config")
            if isinstance(nested, dict) and isinstance(nested.get("keras_history"), list):
                return nested["keras_history"]
        return None

    for layer in layers:
        inbound = layer.get("inbound_nodes")
        if not isinstance(inbound, list):
            continue
        if inbound and all(isinstance(item, list) for item in inbound):
            continue

        converted = []
        for node in inbound:
            connections = []
            if isinstance(node, dict) and isinstance(node.get("args"), list):
                for argument in node["args"]:
                    history = get_history(argument)
                    if history and len(history) >= 3:
                        layer_name, node_index, tensor_index = history[:3]
                        connections.append([layer_name, node_index, tensor_index, {}])
            converted.append(connections)
        layer["inbound_nodes"] = converted

    input_layers = config.get("input_layers")
    if isinstance(input_layers, list) and len(input_layers) == 3 and isinstance(input_layers[0], str):
        config["input_layers"] = [input_layers]
    output_layers = config.get("output_layers")
    if isinstance(output_layers, list) and len(output_layers) == 3 and isinstance(output_layers[0], str):
        config["output_layers"] = [output_layers]

    path.write_text(json.dumps(data), encoding="utf-8")


def smoke_check_tfjs_json(path: pathlib.Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data["modelTopology"]["model_config"]["config"]
    assert isinstance(config["input_layers"][0], list), "input_layers not nested"
    assert isinstance(config["output_layers"][0], list), "output_layers not nested"

    manifest = data.get("weightsManifest")
    assert isinstance(manifest, list) and manifest, "weightsManifest missing"
    for group in manifest:
        paths = group.get("paths")
        weights = group.get("weights")
        assert isinstance(paths, list) and paths, "weightsManifest paths missing"
        assert isinstance(weights, list) and weights, "weightsManifest weights missing"
        for relative_path in paths:
            assert (path.parent / relative_path).exists(), f"missing weight shard {relative_path}"

    for layer in config["layers"]:
        inbound = layer.get("inbound_nodes", [])
        if not isinstance(inbound, list) or any(not isinstance(node, list) for node in inbound):
            raise AssertionError("inbound_nodes contains invalid data")
        if layer.get("class_name") == "InputLayer":
            assert "batchInputShape" in layer.get("config", {}), "InputLayer missing batchInputShape"


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
    extra_metrics: dict | None = None,
):
    records = read_json_list(TRAINING_HISTORY)
    record = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "policy_version": POLICY_VERSION,
        "feature_planes": PLANES,
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
    if extra_metrics:
        record.update(extra_metrics)

    records.append(record)
    write_json(TRAINING_HISTORY, records[-365:])


def split_arrays(*arrays, train_ratio=0.9):
    count = len(arrays[0])
    split = max(1, int(train_ratio * count))
    if split >= count:
        split = count - 1 if count > 1 else count
    return [(array[:split], array[split:]) for array in arrays]


def main():
    X, policy_y, value_y, policy_weights, value_weights, self_play_count, fresh_count, stockfish_count = load_dataset()
    X = ensure_4d_board(X)
    (Xtr, Xva), (Ptr, Pva), (Vtr, Vva), (PWtr, PWva), (VWtr, VWva) = split_arrays(
        X, policy_y, value_y, policy_weights, value_weights
    )

    print(f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, self-play {self_play_count}, Stockfish replay {stockfish_count}")
    baseline_model = load_saved_dual_head_model(CONTINUE_LR, quiet=True)
    model, resumed = load_or_build_model(X.shape[1:])
    epochs = CONTINUE_EPOCHS if resumed else COLD_START_EPOCHS
    baseline_eval = evaluate_model(baseline_model, Xva, Pva, Vva, PWva, VWva, "previous")

    history = model.fit(
        Xtr,
        [Ptr, Vtr],
        validation_data=(Xva, [Pva, Vva], [PWva, VWva]),
        sample_weight=[PWtr, VWtr],
        epochs=epochs,
        batch_size=256,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
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
