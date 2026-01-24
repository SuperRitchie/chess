# ml/train.py
"""
Train a small TF model to regress Stockfish centipawn evals.
Saves TFJS model at public/nn/model.json (+ weights) and patches model.json
to be tfjs-layers compatible (Keras 3 -> legacy node format).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter logs in CI

import json
import pathlib
import random
import numpy as np
import tensorflow as tf
from features import board_to_features

LABELS = pathlib.Path("ml/data/labels.json")
OUT_DIR = pathlib.Path("public/nn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOARD_H, BOARD_W, PLANES = 8, 8, 13
FLAT_SIZE = BOARD_H * BOARD_W * PLANES  # 832


def load_dataset():
    items = json.loads(LABELS.read_text(encoding="utf-8"))
    random.shuffle(items)
    X, y = [], []
    for it in items:
        x = board_to_features(it["fen"])  # may be (8,8,13) OR flattened (832,)
        X.append(x)
        # Clip extreme mates to +/-2000 cp to help regression stability
        y.append(np.clip(float(it["cp"]), -2000.0, 2000.0))
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def ensure_4d_board(X: np.ndarray) -> np.ndarray:
    """
    Ensure X has shape (N, 8, 8, 13). If X is (N, 832) (flattened), reshape it.
    """
    if X.ndim == 4 and X.shape[1:] == (BOARD_H, BOARD_W, PLANES):
        return X
    if X.ndim == 2 and X.shape[1] == FLAT_SIZE:
        return X.reshape(-1, BOARD_H, BOARD_W, PLANES)
    raise ValueError(
        f"Expected X as (N, {BOARD_H}, {BOARD_W}, {PLANES}) or (N, {FLAT_SIZE}), "
        f"but got {X.shape}. Check features.board_to_features."
    )


def build_model(input_shape=(BOARD_H, BOARD_W, PLANES)):
    inp = tf.keras.Input(shape=input_shape, name="board")
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="cp")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return model


def patch_tfjs_model_json(path: pathlib.Path) -> None:
    """
    Patch Keras 3 style TFJS model.json to tfjs-layers compatible format:
      - InputLayer: batch_shape -> batchInputShape
      - inbound_nodes: object nodes -> legacy array nodes
      - input_layers/output_layers: flat -> nested arrays
    """
    j = json.loads(path.read_text(encoding="utf-8"))

    cfg = (
        j.get("modelTopology", {})
         .get("model_config", {})
         .get("config", {})
    )
    layers = cfg.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("model.json missing modelTopology.model_config.config.layers")

    # 1) InputLayer key
    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            c = layer.get("config", {})
            if "batch_shape" in c and "batchInputShape" not in c:
                c["batchInputShape"] = c.pop("batch_shape")

    def get_history(arg):
        # Accept either {"keras_history":[...]} or {"config":{"keras_history":[...]}}
        if isinstance(arg, dict):
            if isinstance(arg.get("keras_history"), list):
                return arg["keras_history"]
            c2 = arg.get("config")
            if isinstance(c2, dict) and isinstance(c2.get("keras_history"), list):
                return c2["keras_history"]
        return None

    # 2) inbound_nodes conversion
    for layer in layers:
        inbound = layer.get("inbound_nodes")
        if not isinstance(inbound, list):
            continue

        # already legacy? (array-of-arrays)
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

    # 3) input_layers/output_layers nesting
    il = cfg.get("input_layers")
    if isinstance(il, list) and len(il) == 3 and isinstance(il[0], str):
        cfg["input_layers"] = [il]
    ol = cfg.get("output_layers")
    if isinstance(ol, list) and len(ol) == 3 and isinstance(ol[0], str):
        cfg["output_layers"] = [ol]

    path.write_text(json.dumps(j), encoding="utf-8")


def smoke_check_tfjs_json(path: pathlib.Path) -> None:
    """
    Fast structural checks so CI fails early if the JSON will break tfjs-layers.
    (Doesn't require Node or a browser.)
    """
    j = json.loads(path.read_text(encoding="utf-8"))
    cfg = j["modelTopology"]["model_config"]["config"]

    # input/output layers must be nested arrays
    il = cfg["input_layers"]
    ol = cfg["output_layers"]
    assert isinstance(il, list) and len(il) > 0 and isinstance(il[0], list), "input_layers not nested"
    assert isinstance(ol, list) and len(ol) > 0 and isinstance(ol[0], list), "output_layers not nested"

    # inbound_nodes entries must be arrays (legacy), not dict objects
    for layer in cfg["layers"]:
        inbound = layer.get("inbound_nodes", [])
        if not isinstance(inbound, list):
            raise AssertionError("inbound_nodes missing/invalid")
        for node in inbound:
            if not isinstance(node, list):
                raise AssertionError("inbound_nodes contains non-list node (likely object style)")

    # InputLayer should have batchInputShape
    for layer in cfg["layers"]:
        if layer.get("class_name") == "InputLayer":
            c = layer.get("config", {})
            assert "batchInputShape" in c, "InputLayer missing batchInputShape"


def main():
    X, y = load_dataset()
    X = ensure_4d_board(X)

    n = len(X)
    split = int(0.9 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    print(f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, ytr {ytr.shape}, yva {yva.shape}")

    model = build_model(X.shape[1:])

    model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=20,
        batch_size=512,
        verbose=2,
    )

    # Save TFJS (direct Keras -> TFJS)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, str(OUT_DIR))

    # Patch + validate
    model_json = OUT_DIR / "model.json"
    patch_tfjs_model_json(model_json)
    smoke_check_tfjs_json(model_json)

    print("Saved TFJS model to", model_json)


if __name__ == "__main__":
    main()
