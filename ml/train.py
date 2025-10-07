# ml/train.py
"""
Train a small TF model to regress Stockfish centipawn evals.
Saves TFJS model at public/nn/model.json (+ weights).
"""
import json, pathlib, random, numpy as np, tensorflow as tf
from features import board_to_features

LABELS = pathlib.Path("ml/data/labels.json")
OUT_DIR = pathlib.Path("public/nn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    items = json.loads(LABELS.read_text(encoding="utf-8"))
    random.shuffle(items)
    X, y = [], []
    for it in items:
        x = board_to_features(it["fen"])
        X.append(x)
        # Clip extreme mates to +/-2000 cp to help regression stability
        y.append(np.clip(float(it["cp"]), -2000.0, 2000.0))
    X = np.stack(X).astype(np.float32)  # expect shape (N, 8, 8, 13)
    y = np.array(y, dtype=np.float32)
    return X, y

def build_model(input_shape=(8, 8, 13)):
    inp = tf.keras.Input(shape=input_shape, name="board")
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="cp")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="huber",
    )
    return model

def main():
    X, y = load_dataset()
    n = len(X)
    split = int(0.9 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    # Fix: use the full input shape (8,8,13), not a single int
    model = build_model(X.shape[1:])

    model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=6,
        batch_size=512,
        verbose=2,
    )

    # Save TFJS
    tf.saved_model.save(model, "ml/tmp_savedmodel")
    import tensorflowjs as tfjs
    tfjs.converters.convert_tf_saved_model("ml/tmp_savedmodel", str(OUT_DIR))
    print("Saved TFJS model to", OUT_DIR)

if __name__ == "__main__":
    main()
