# ml/self_play.py
"""
generate AlphaZero-style self-play samples using policy-guided MCTS
"""
import json
import math
import os
import pathlib
import random

import chess
import numpy as np
import tensorflow as tf

from features import board_to_features
from policy_map import POLICY_SIZE, move_to_index, to_queen_promotion

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
SEED = int(os.environ.get("AZ_SELF_PLAY_SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    total = np.sum(e)
    if total <= 0:
        return np.ones_like(x, dtype=np.float32) / len(x)
    return e / total


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


def load_model_or_none():
    if not CHECKPOINT_MODEL.exists():
        print("[self-play] no checkpoint yet, using uniform priors")
        return None
    try:
        model = tf.keras.models.load_model(CHECKPOINT_MODEL, compile=False)
    except Exception as exc:
        print(f"[self-play] checkpoint could not be loaded, using uniform priors: {exc}")
        return None
    if len(model.outputs) != 2:
        print("[self-play] checkpoint is value-only, using uniform priors until dual-head model exists")
        return None
    return model


def terminal_value(board):
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def model_policy_value(model, board):
    legal_moves = [to_queen_promotion(move) for move in board.legal_moves]
    if not legal_moves:
        return {}, terminal_value(board) or 0.0

    if model is None:
        p = 1.0 / len(legal_moves)
        return {move_to_index(move): p for move in legal_moves}, 0.0

    x = board_to_features(board.fen(en_passant="fen")).reshape(1, 8, 8, 13).astype(np.float32)
    prediction = model.predict(x, verbose=0)
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
        p = 1.0 / len(legal_moves)
        return {move_to_index(move): p for move in legal_moves}, 0.0

    policy_logits, value = prediction
    logits = policy_logits[0]
    legal_indices = [move_to_index(move) for move in legal_moves]
    legal_logits = np.array([logits[idx] for idx in legal_indices], dtype=np.float32)
    legal_probs = softmax(legal_logits)
    priors = {idx: float(prob) for idx, prob in zip(legal_indices, legal_probs)}
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
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self):
        best_score = -float("inf")
        best_child = None
        parent_visits = max(1, self.visit_count)
        for child in self.children.values():
            q = 0.0 if child.visit_count == 0 else -child.value
            u = CPUCT * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, priors):
        for move in self.board.legal_moves:
            normalized = to_queen_promotion(move)
            idx = move_to_index(normalized)
            next_board = self.board.copy(stack=False)
            if normalized in next_board.legal_moves:
                next_board.push(normalized)
            else:
                next_board.push(move)
            self.children[idx] = Node(next_board, parent=self, prior=priors.get(idx, 0.0), move=normalized)

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
    for child, n in zip(children, noise):
        child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * float(n)


def run_search(model, board):
    root = Node(board.copy(stack=False))
    priors, value = model_policy_value(model, root.board)
    root.expand(priors)
    add_root_noise(root)
    root.backup(value)

    for _ in range(MCTS_SEARCHES):
        node = root
        while node.children:
            node = node.select_child()

        value = terminal_value(node.board)
        if value is None:
            priors, value = model_policy_value(model, node.board)
            node.expand(priors)
        node.backup(value)

    visits = {idx: child.visit_count for idx, child in root.children.items()}
    total = sum(visits.values())
    if total == 0:
        legal = [move_to_index(to_queen_promotion(move)) for move in board.legal_moves]
        p = 1.0 / max(1, len(legal))
        return {idx: p for idx in legal}
    return {idx: count / total for idx, count in visits.items()}


def choose_action(policy, move_number):
    indices = list(policy.keys())
    probs = np.array([policy[idx] for idx in indices], dtype=np.float32)
    probs = probs / np.sum(probs)
    if move_number < TEMP_MOVES:
        return int(np.random.choice(indices, p=probs))
    return int(indices[int(np.argmax(probs))])


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
        policy = run_search(model, board)
        action = choose_action(policy, ply)
        legal_by_index = {move_to_index(to_queen_promotion(move)): to_queen_promotion(move) for move in board.legal_moves}
        move = legal_by_index.get(action)
        if move is None:
            move = random.choice(list(board.legal_moves))
            action = move_to_index(to_queen_promotion(move))

        samples.append({
            "fen": board.fen(en_passant="fen"),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "policy": [[int(idx), float(prob)] for idx, prob in policy.items() if prob > 0],
        })
        board.push(move)

    white_result = result_for_white(board)
    for sample in samples:
        sample["z"] = white_result if sample["turn"] == "white" else -white_result

    print(f"[self-play] game {game_index + 1}: {len(samples)} plies, result {board.result(claim_draw=True)}")
    return samples


def main():
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
