# ml/train_fixed_eval.py
"""train with persistent neural and search-quality holdouts"""
import os
import random

import chess
import numpy as np
import tensorflow as tf

import train
from self_play import arena_score, run_search_batch

FIXED_EVAL_SET = train.pathlib.Path("ml/data/fixed_eval_set_v3.json")
FIXED_EVAL_SELF = int(os.environ.get("AZ_FIXED_EVAL_SELF", "512"))
FIXED_EVAL_STOCKFISH = int(os.environ.get("AZ_FIXED_EVAL_STOCKFISH", "512"))
ARENA_GAMES = int(os.environ.get("AZ_ARENA_GAMES", "2"))
ARENA_SEARCHES = int(os.environ.get("AZ_ARENA_SEARCHES", "24"))
ARENA_MAX_PLIES = int(os.environ.get("AZ_ARENA_MAX_PLIES", "160"))
ARENA_MIN_SCORE = float(os.environ.get("AZ_ARENA_MIN_SCORE", "0.5"))
MCTS_EVAL_POSITIONS = int(os.environ.get("AZ_MCTS_EVAL_POSITIONS", "48"))
MCTS_EVAL_SEARCHES = int(os.environ.get("AZ_MCTS_EVAL_SEARCHES", "64"))
MIN_MCTS_ALIGNMENT_IMPROVEMENT = float(os.environ.get("AZ_MIN_MCTS_ALIGNMENT_IMPROVEMENT", "0.0001"))
MIN_MCTS_TOP_MOVE_IMPROVEMENT = float(os.environ.get("AZ_MIN_MCTS_TOP_MOVE_IMPROVEMENT", "0.0"))
_EPS = 1e-7


def _valid_self_play_sample(item: dict) -> dict | None:
    fen = item.get("fen")
    if not fen:
        return None
    policy_version = int(item.get("policy_version", 1))
    policy = train.dense_policy_from_sparse(item.get("policy"), fen=fen, policy_version=policy_version)
    if float(np.sum(policy)) <= 0:
        return None
    try:
        outcome = float(item.get("z"))
    except (TypeError, ValueError):
        return None
    return {
        "source": "self_play",
        "fen": fen,
        "policy_version": policy_version,
        "policy": item.get("policy"),
        "z": float(np.clip(outcome, -1.0, 1.0)),
    }


def _valid_stockfish_sample(item: dict) -> dict | None:
    fen = item.get("fen")
    if not fen:
        return None
    try:
        cp = float(item.get("cp", 0.0))
    except (TypeError, ValueError):
        return None
    policy_version = int(item.get("policy_version", 1))
    policy = train.dense_policy_from_sparse(item.get("policy"), fen=fen, policy_version=policy_version)
    if float(np.sum(policy)) <= 0:
        return None
    return {
        "source": "stockfish",
        "fen": fen,
        "cp": cp,
        "policy_version": policy_version,
        "policy": item.get("policy"),
    }


def _pick_deterministic(items: list[dict], count: int, salt: str) -> list[dict]:
    rng = random.Random(f"{train.TRAIN_SEED}:{salt}")
    selected = list(items)
    rng.shuffle(selected)
    return selected[:count]


def create_fixed_eval_set() -> list[dict]:
    self_items = []
    for item in train.read_json_list(train.SELF_PLAY_BUFFER):
        sample = _valid_self_play_sample(item)
        if sample is not None:
            self_items.append(sample)

    stockfish_by_fen = {}
    source_items = train.read_json_list(train.STOCKFISH_REPLAY_BUFFER) + train.read_json_list(train.LABELS)
    for item in train.normalize_labels(source_items):
        sample = _valid_stockfish_sample(item)
        if sample is not None:
            stockfish_by_fen[sample["fen"]] = sample

    fixed_items = _pick_deterministic(self_items, FIXED_EVAL_SELF, "self-v2")
    fixed_items += _pick_deterministic(list(stockfish_by_fen.values()), FIXED_EVAL_STOCKFISH, "stockfish-v3")
    if not fixed_items:
        raise ValueError("could not create fixed evaluation set: no valid samples")

    fixed_items = _pick_deterministic(fixed_items, len(fixed_items), "mixed-v2")
    train.write_json(FIXED_EVAL_SET, fixed_items)
    print(f"[train] created persistent holdout at {FIXED_EVAL_SET}")
    return fixed_items


def load_fixed_eval_set() -> list[dict]:
    items = train.read_json_list(FIXED_EVAL_SET)
    stockfish_count = sum(item.get("source") == "stockfish" for item in items)
    return items if items and stockfish_count > 0 else create_fixed_eval_set()


def fixed_eval_arrays(samples: list[dict]):
    X = []
    policy_y = []
    value_y = []
    policy_weights = []
    value_weights = []
    self_count = 0
    stockfish_count = 0
    for sample in samples:
        source = sample.get("source")
        fen = sample.get("fen")
        if not fen:
            continue

        if source == "self_play":
            policy = train.dense_policy_from_sparse(
                sample.get("policy"),
                fen=fen,
                policy_version=int(sample.get("policy_version", 1)),
            )
            if float(np.sum(policy)) <= 0:
                continue
            try:
                value = float(sample.get("z"))
            except (TypeError, ValueError):
                continue
            X.append(train.board_to_features(fen))
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
            policy = train.dense_policy_from_sparse(
                sample.get("policy"),
                fen=fen,
                policy_version=int(sample.get("policy_version", train.POLICY_VERSION)),
            )
            if float(np.sum(policy)) <= 0:
                continue
            X.append(train.board_to_features(fen))
            policy_y.append(policy)
            value_y.append(train.cp_to_value(cp))
            policy_weights.append(1.0)
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
        f"[train] fixed holdout {len(X_arr)} positions "
        f"(self-play {self_count}, Stockfish {stockfish_count})"
    )
    return X_arr, P_arr, V_arr, PW_arr, VW_arr


def _softmax_cross_entropy(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float32) - np.max(logits, axis=1, keepdims=True)
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
    value_prediction = np.asarray(predictions[1], dtype=np.float32).reshape(-1, 1)
    if policy_logits.shape != P.shape or value_prediction.shape != V.shape:
        print(
            f"[train] could not evaluate {label} model: output shapes "
            f"{policy_logits.shape}/{value_prediction.shape} do not match {P.shape}/{V.shape}"
        )
        return None

    policy_denominator = max(float(np.sum(PW)), _EPS)
    value_denominator = max(float(np.sum(VW)), _EPS)
    policy_loss = float(np.sum(_softmax_cross_entropy(P, policy_logits) * PW) / policy_denominator)
    value_error = (V - value_prediction).reshape(-1)
    value_loss = float(np.sum(np.square(value_error) * VW) / value_denominator)
    value_mae = float(np.sum(np.abs(value_error) * VW) / value_denominator)
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


def evaluate_mcts_alignment(model: tf.keras.Model | None, samples: list[dict], label: str) -> dict | None:
    if model is None or MCTS_EVAL_POSITIONS <= 0:
        return None

    boards = []
    targets = []
    for sample in samples:
        if sample.get("source") != "stockfish":
            continue
        fen = sample.get("fen")
        if not fen:
            continue
        target = train.dense_policy_from_sparse(
            sample.get("policy"),
            fen=fen,
            policy_version=int(sample.get("policy_version", train.POLICY_VERSION)),
        )
        if float(np.sum(target)) <= 0:
            continue
        try:
            boards.append(chess.Board(fen))
        except ValueError:
            continue
        targets.append(target)
        if len(boards) >= MCTS_EVAL_POSITIONS:
            break

    if not boards:
        print(f"[mcts-eval] {label} unavailable: no Stockfish policy holdout")
        return None

    search_policies = run_search_batch(
        model,
        boards,
        searches=MCTS_EVAL_SEARCHES,
        add_noise=False,
    )
    alignments = []
    top_move_hits = []
    for search_policy, target in zip(search_policies, targets):
        alignments.append(sum(float(target[index]) * probability for index, probability in search_policy.items()))
        chosen = max(search_policy, key=search_policy.get)
        top_move_hits.append(chosen == int(np.argmax(target)))

    metrics = {
        "positions": len(boards),
        "searches": MCTS_EVAL_SEARCHES,
        "alignment": float(np.mean(alignments)),
        "top_move_accuracy": float(np.mean(top_move_hits)),
    }
    print(
        f"[mcts-eval] {label} alignment {metrics['alignment']:.6f}, "
        f"top move {metrics['top_move_accuracy']:.3f} over {len(boards)} positions"
    )
    return metrics


def mcts_candidate_passes(
    baseline_metrics: dict | None,
    candidate_metrics: dict | None,
) -> tuple[bool, str]:
    if baseline_metrics is None or candidate_metrics is None:
        return False, "candidate_mcts_evaluation_unavailable"

    baseline_alignment = baseline_metrics.get("alignment")
    candidate_alignment = candidate_metrics.get("alignment")
    baseline_accuracy = baseline_metrics.get("top_move_accuracy")
    candidate_accuracy = candidate_metrics.get("top_move_accuracy")
    values = [baseline_alignment, candidate_alignment, baseline_accuracy, candidate_accuracy]
    if any(value is None or not np.isfinite(value) for value in values):
        return False, "candidate_mcts_metrics_unavailable"

    required_alignment = baseline_alignment + MIN_MCTS_ALIGNMENT_IMPROVEMENT
    if candidate_alignment < required_alignment:
        return False, "candidate_mcts_alignment_did_not_improve"

    required_accuracy = baseline_accuracy + MIN_MCTS_TOP_MOVE_IMPROVEMENT
    if candidate_accuracy < required_accuracy:
        return False, "candidate_mcts_top_move_accuracy_regressed"

    return True, "mcts_policy_improved"


def main():
    fixed_samples = load_fixed_eval_set()
    excluded_fens = {item.get("fen") for item in fixed_samples if item.get("fen")}
    Xev, Pev, Vev, PWev, VWev = fixed_eval_arrays(fixed_samples)

    X, policy_y, value_y, policy_weights, value_weights, self_play_count, fresh_count, stockfish_count = train.load_dataset(
        excluded_fens=excluded_fens
    )
    X = train.ensure_4d_board(X)
    (Xtr, Xva), (Ptr, Pva), (Vtr, Vva), (PWtr, PWva), (VWtr, VWva) = train.split_arrays(
        X, policy_y, value_y, policy_weights, value_weights
    )

    print(
        f"[train] Xtr {Xtr.shape}, Xva {Xva.shape}, self-play {self_play_count}, "
        f"Stockfish replay {stockfish_count}; excluded {len(excluded_fens)} holdout FENs"
    )

    baseline_model = train.load_saved_dual_head_model(train.CONTINUE_LR, quiet=True)
    model, resumed = train.load_or_build_model(X.shape[1:])
    epochs = train.CONTINUE_EPOCHS if resumed else train.COLD_START_EPOCHS

    moving_baseline_eval = train.evaluate_model(baseline_model, Xva, Pva, Vva, PWva, VWva, "previous moving")
    fixed_baseline_eval = evaluate_fixed_model(baseline_model, Xev, Pev, Vev, PWev, VWev, "previous fixed")

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

    moving_candidate_eval = train.evaluate_model(model, Xva, Pva, Vva, PWva, VWva, "candidate moving")
    fixed_candidate_eval = evaluate_fixed_model(model, Xev, Pev, Vev, PWev, VWev, "candidate fixed")
    accepted, gate_reason = train.should_accept_candidate(fixed_candidate_eval, fixed_baseline_eval, resumed)

    baseline_mcts_eval = None
    candidate_mcts_eval = None
    if accepted and resumed:
        baseline_mcts_eval = evaluate_mcts_alignment(baseline_model, fixed_samples, "previous")
        candidate_mcts_eval = evaluate_mcts_alignment(model, fixed_samples, "candidate")
        mcts_accepted, mcts_reason = mcts_candidate_passes(
            baseline_mcts_eval,
            candidate_mcts_eval,
        )
        if not mcts_accepted:
            accepted = False
            gate_reason = mcts_reason
        else:
            gate_reason = f"{gate_reason}_and_{mcts_reason}"

    arena_result = None
    if accepted and resumed and baseline_model is not None and ARENA_GAMES > 0:
        arena_result = arena_score(
            model,
            baseline_model,
            games=ARENA_GAMES,
            searches=ARENA_SEARCHES,
            max_plies=ARENA_MAX_PLIES,
        )
        print(f"[arena] candidate mean score {arena_result:.3f}; required {ARENA_MIN_SCORE:.3f}")
        if arena_result < ARENA_MIN_SCORE:
            accepted = False
            gate_reason = "candidate_arena_score_too_low"
        else:
            gate_reason = f"{gate_reason}_and_arena_passed"

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

    extra_metrics = {
        "fixed_holdout_positions": len(Xev),
        "fixed_holdout_excluded_from_training": True,
    }
    if arena_result is not None:
        extra_metrics.update(
            {
                "arena_games": ARENA_GAMES,
                "arena_searches": ARENA_SEARCHES,
                "arena_candidate_score": arena_result,
                "arena_min_score": ARENA_MIN_SCORE,
            }
        )
    if baseline_mcts_eval and candidate_mcts_eval:
        extra_metrics.update(
            {
                "mcts_eval_positions": candidate_mcts_eval["positions"],
                "mcts_eval_searches": candidate_mcts_eval["searches"],
                "baseline_mcts_alignment": baseline_mcts_eval["alignment"],
                "candidate_mcts_alignment": candidate_mcts_eval["alignment"],
                "baseline_mcts_top_move_accuracy": baseline_mcts_eval["top_move_accuracy"],
                "candidate_mcts_top_move_accuracy": candidate_mcts_eval["top_move_accuracy"],
                "min_mcts_alignment_improvement": MIN_MCTS_ALIGNMENT_IMPROVEMENT,
                "min_mcts_top_move_improvement": MIN_MCTS_TOP_MOVE_IMPROVEMENT,
            }
        )

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
        extra_metrics=extra_metrics,
    )

    if moving_baseline_eval and moving_candidate_eval:
        print(
            "[train] moving validation was diagnostic only: "
            f"previous={moving_baseline_eval.get('loss'):.6f}, "
            f"candidate={moving_candidate_eval.get('loss'):.6f}"
        )


if __name__ == "__main__":
    main()
