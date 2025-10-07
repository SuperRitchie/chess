# ml/train.py
"""
Train a small TF model to regress Stockfish centipawn evals.
Saves TFJS model at public/nn/model.json (+ weights).
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


def main():
    X, y = load_dataset()
    # Auto-fix shape if features are flattened
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
        epochs=6,
        batch_size=512,
        verbose=2,
    )

    # Save TFJS (direct Keras -> TFJS)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, str(OUT_DIR))
    print("Saved TFJS model to", OUT_DIR / "model.json")


if __name__ == "__main__":
    main()
