# ml/self_play.py
"""Generate AlphaZero-style self-play samples with neural PUCT search."""
import json
import math
import os
import pathlib
import random

import chess
import numpy as np
import tensorflow as tf

from features import PLANES, board_to_features
from policy_map import POLICY_SIZE, POLICY_VERSION, move_to_index

CHECKPOINT_MODEL = pathlib.Path("ml/checkpoints/chess_eval.keras")
SELF_PLAY_BUFFER = pathlib.Path("ml/data/self_play_buffer.json")

SELF_PLAY_GAMES = int(os.environ.get("AZ_SELF_PLAY_GAMES", "4"))
MCTS_SEARCHES = int(os.environ.get("AZ_MCTS_SEARCHES", "80"))
MAX_PLIES = int(os.environ.get("AZ_MAX_PLIES", "180"))
MAX_BUFFER = int(os.environ.get("AZ_MAX_SELF_PLAY_SAMPLES", "8000"))
CPUCT = float(os.environ.get("AZ_CPUCT", "1.5"))
DIRICHLET_ALPHA = float(os.environ.get("AZ_DIRICHLET_ALPHA", "0.3"))
DIRICHLET_EPSILON = float(os.environ.get("AZ_DIRICHLET_EPSILON", "0.25"))
TEMP_MOVES = int(os.environ.get("AZ_TEMP_MOVES", "20"))
TEMPERATURE = float(os.environ.get("AZ_TEMPERATURE", "1.0"))
SEED = int(os.environ.get("AZ_SELF_PLAY_SEED", "42")) % (2**32 - 1)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exponentials = np.exp(values)
    total = float(np.sum(exponentials))
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(values, dtype=np.float32) / max(1, len(values))
    return exponentials / total


def read_json_list(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def is_compatible_model(model) -> bool:
    try:
        input_shape = tuple(model.input_shape[1:])
        policy_size = int(model.outputs[0].shape[-1])
        value_size = int(model.outputs[1].shape[-1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return False
    return len(model.outputs) == 2 and input_shape == (8, 8, PLANES) and policy_size == POLICY_SIZE and value_size == 1


def load_model_or_none(path=CHECKPOINT_MODEL):
    if not path.exists():
        print("[self-play] no checkpoint yet, using uniform priors")
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        print(f"[self-play] checkpoint could not be loaded, using uniform priors: {exc}")
        return None
    if not is_compatible_model(model):
        print("[self-play] checkpoint shape is obsolete, using uniform priors until the migrated model is saved")
        return None
    return model


def terminal_value(board):
    """Return a terminal value from the side-to-move perspective."""
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def model_policy_value(model, board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, terminal_value(board) or 0.0

    if model is None:
        probability = 1.0 / len(legal_moves)
        return {move_to_index(move): probability for move in legal_moves}, 0.0

    features = board_to_features(board.fen(en_passant="fen")).reshape(1, 8, 8, PLANES).astype(np.float32)
    prediction = model.predict(features, verbose=0)
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
        probability = 1.0 / len(legal_moves)
        return {move_to_index(move): probability for move in legal_moves}, 0.0

    policy_logits, value = prediction
    logits = np.asarray(policy_logits[0], dtype=np.float32)
    legal_indices = [move_to_index(move) for move in legal_moves]
    legal_logits = np.array([logits[index] for index in legal_indices], dtype=np.float32)
    legal_probabilities = softmax(legal_logits)
    priors = {index: float(probability) for index, probability in zip(legal_indices, legal_probabilities)}
    return priors, float(value[0][0])


class Node:
    def __init__(self, board, parent=None, prior=0.0, move=None):
        self.board = board
        self.parent = parent
        self.prior = float(prior)
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def select_child(self):
        best_score = -float("inf")
        best_child = None
        parent_visits = max(1, self.visit_count)
        for child in self.children.values():
            q_value = 0.0 if child.visit_count == 0 else -child.value
            exploration = CPUCT * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, priors):
        for move in self.board.legal_moves:
            index = move_to_index(move)
            if index in self.children:
                continue
            next_board = self.board.copy(stack=True)
            next_board.push(move)
            self.children[index] = Node(next_board, parent=self, prior=priors.get(index, 0.0), move=move)

    def backup(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent


def add_root_noise(root):
    children = list(root.children.values())
    if not children:
        return
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(children))
    for child, sample in zip(children, noise):
        child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * float(sample)


def run_search(model, board, searches=None, add_noise=True):
    searches = MCTS_SEARCHES if searches is None else int(searches)
    root = Node(board.copy(stack=True))
    priors, root_value = model_policy_value(model, root.board)
    root.expand(priors)
    if add_noise:
        add_root_noise(root)
    root.visit_count = 1
    root.value_sum = root_value

    for _ in range(max(1, searches)):
        node = root
        while node.children:
            node = node.select_child()
            if node is None:
                break
        if node is None:
            break

        value = terminal_value(node.board)
        if value is None:
            priors, value = model_policy_value(model, node.board)
            node.expand(priors)
        node.backup(value)

    visits = {index: child.visit_count for index, child in root.children.items()}
    total = sum(visits.values())
    if total <= 0:
        legal_indices = [move_to_index(move) for move in board.legal_moves]
        probability = 1.0 / max(1, len(legal_indices))
        return {index: probability for index in legal_indices}
    return {index: count / total for index, count in visits.items()}


def choose_action(policy, move_number, sample=True):
    indices = list(policy.keys())
    if not indices:
        raise ValueError("cannot choose from an empty policy")
    probabilities = np.array([policy[index] for index in indices], dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)

    if sample and move_number < TEMP_MOVES and TEMPERATURE > 0:
        probabilities = np.power(probabilities, 1.0 / TEMPERATURE)
        probabilities = probabilities / np.sum(probabilities)
        return int(np.random.choice(indices, p=probabilities))
    return int(indices[int(np.argmax(probabilities))])


def result_for_white(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


def play_game(model, game_index):
    board = chess.Board()
    samples = []

    for ply in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break
        policy = run_search(model, board, add_noise=True)
        action = choose_action(policy, ply, sample=True)
        legal_by_index = {move_to_index(move): move for move in board.legal_moves}
        move = legal_by_index.get(action)
        if move is None:
            move = random.choice(list(board.legal_moves))

        samples.append(
            {
                "fen": board.fen(en_passant="fen"),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "policy_version": POLICY_VERSION,
                "policy": [[int(index), float(probability)] for index, probability in policy.items() if probability > 0],
            }
        )
        board.push(move)

    white_result = result_for_white(board)
    for sample in samples:
        sample["z"] = white_result if sample["turn"] == "white" else -white_result

    print(f"[self-play] game {game_index + 1}: {len(samples)} plies, result {board.result(claim_draw=True)}")
    return samples


def play_arena_game(candidate, baseline, candidate_is_white, searches=24, max_plies=160, opening=None):
    board = chess.Board()
    for uci in opening or []:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"invalid arena opening move {uci}")
        board.push(move)

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        candidate_turn = board.turn == (chess.WHITE if candidate_is_white else chess.BLACK)
        model = candidate if candidate_turn else baseline
        policy = run_search(model, board, searches=searches, add_noise=False)
        action = choose_action(policy, ply, sample=False)
        move = {move_to_index(move): move for move in board.legal_moves}.get(action)
        if move is None:
            return 0.0
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    candidate_color = chess.WHITE if candidate_is_white else chess.BLACK
    return 1.0 if outcome.winner == candidate_color else 0.0


def arena_score(candidate, baseline, games=2, searches=24, max_plies=160):
    openings = (("e2e4", "e7e5"), ("d2d4", "d7d5"), ("c2c4", "e7e5"))
    scores = []
    for game_index in range(max(0, games)):
        candidate_is_white = game_index % 2 == 0
        opening = openings[(game_index // 2) % len(openings)]
        score = play_arena_game(
            candidate,
            baseline,
            candidate_is_white,
            searches=searches,
            max_plies=max_plies,
            opening=opening,
        )
        scores.append(score)
        print(f"[arena] game {game_index + 1}/{games}: candidate score {score:.1f}")
    return float(np.mean(scores)) if scores else 1.0


def main():
    seed_everything()
    model = load_model_or_none()
    existing = read_json_list(SELF_PLAY_BUFFER)
    new_samples = []

    for game_index in range(SELF_PLAY_GAMES):
        new_samples.extend(play_game(model, game_index))

    merged = existing + new_samples
    if len(merged) > MAX_BUFFER:
        merged = merged[-MAX_BUFFER:]

    write_json(SELF_PLAY_BUFFER, merged)
    print(f"[self-play] saved {len(new_samples)} new samples, buffer now {len(merged)}")


if __name__ == "__main__":
    main()
