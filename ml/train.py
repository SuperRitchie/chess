# ml/train.py
"""
train the chess evaluator and keep a saved brain for the next run
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

LABELS = pathlib.Path("ml/data/labels.json")
REPLAY_BUFFER = pathlib.Path("ml/data/replay_buffer.json")
TRAINING_HISTORY = pathlib.Path("ml/training_history.json")
CHECKPOINT_DIR = pathlib.Path("ml/checkpoints")
CHECKPOINT_MODEL = CHECKPOINT_DIR / "chess_eval.keras"
OUT_DIR = pathlib.Path("public/nn")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BOARD_H, BOARD_W, PLANES = 8, 8, 13
FLAT_SIZE = BOARD_H * BOARD_W * PLANES
MAX_REPLAY_ITEMS = int(os.environ.get("MAX_REPLAY_ITEMS", "50000"))
COLD_START_EPOCHS = int(os.environ.get("COLD_START_EPOCHS", "6"))
CONTINUE_EPOCHS = int(os.environ.get("CONTINUE_EPOCHS", "3"))
COLD_START_LR = float(os.environ.get("COLD_START_LR", "1e-3"))
CONTINUE_LR = float(os.environ.get("CONTINUE_LR", "2e-4"))
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


def merge_replay_buffer(new_items: list[dict]) -> list[dict]:
    existing_items = normalize_labels(read_json_list(REPLAY_BUFFER))
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
    write_json(REPLAY_BUFFER, merged)
    print(f"[train] replay buffer {len(merged)} positions, new {len(new_items)}")
    return merged


def load_dataset():
    fresh_items = normalize_labels(read_json_list(LABELS))
    items = merge_replay_buffer(fresh_items)

    X, y = [], []
    for item in items:
        X.append(board_to_features(item["fen"]))
        y.append(np.clip(float(item["cp"]), -2000.0, 2000.0))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, len(fresh_items), len(items)


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
        loss="huber",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_model(input_shape=(BOARD_H, BOARD_W, PLANES), learning_rate=COLD_START_LR):
    inp = tf.keras.Input(shape=input_shape, name="board")
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="cp")(x)
    model = tf.keras.Model(inp, out)
    return compile_model(model, learning_rate)


def load_or_build_model(input_shape):
    if CHECKPOINT_MODEL.exists():
        print(f"[train] loading saved brain from {CHECKPOINT_MODEL}")
        model = tf.keras.models.load_model(CHECKPOINT_MODEL, compile=False)
        return compile_model(model, CONTINUE_LR), True

    print("[train] no saved brain found, starting from scratch")
    return build_model(input_shape, COLD_START_LR), False


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


def append_training_history(history, resumed, fresh_count, replay_count, epochs):
    records = read_json_list(TRAINING_HISTORY)
    record = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "resumed_from_checkpoint": resumed,
        "fresh_positions": fresh_count,
        "replay_positions": replay_count,
        "epochs": epochs,
    }

    for key, values in history.history.items():
        if values:
            record[f"final_{key}"] = float(values[-1])

    records.append(record)
    write_json(TRAINING_HISTORY, records[-365:])


def main():
    X, y, fresh_count, replay_count = load_dataset()
    X = ensure_4d_board(X)

    n = len(X)
    split = int(0.9 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    print(f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, ytr {ytr.shape}, yva {yva.shape}")

    model, resumed = load_or_build_model(X.shape[1:])
    epochs = CONTINUE_EPOCHS if resumed else COLD_START_EPOCHS

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=512,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    model.save(CHECKPOINT_MODEL)
    print(f"[train] saved brain to {CHECKPOINT_MODEL}")

    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, str(OUT_DIR))

    model_json = OUT_DIR / "model.json"
    patch_tfjs_model_json(model_json)
    smoke_check_tfjs_json(model_json)
    append_training_history(history, resumed, fresh_count, replay_count, epochs)

    print(f"[train] saved TFJS model to {model_json}")


if __name__ == "__main__":
    main()
