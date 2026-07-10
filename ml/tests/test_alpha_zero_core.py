import pathlib
import sys
import unittest

import chess
import numpy as np

ML_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import features
import policy_map
import self_play
import train


class PolicyMapTests(unittest.TestCase):
    def test_promotion_actions_are_unique_and_round_trip(self):
        moves = [
            chess.Move.from_uci("a7a8q"),
            chess.Move.from_uci("a7a8r"),
            chess.Move.from_uci("a7a8b"),
            chess.Move.from_uci("a7a8n"),
        ]
        indices = [policy_map.move_to_index(move) for move in moves]
        self.assertEqual(len(set(indices)), 4)
        self.assertEqual([policy_map.index_to_move(index) for index in indices], moves)

    def test_legacy_promotion_maps_to_queen_channel(self):
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        legacy = chess.A7 * 64 + chess.A8
        normalized = policy_map.normalize_policy_index(legacy, board, policy_version=1)
        self.assertEqual(policy_map.index_to_move(normalized), chess.Move.from_uci("a7a8q"))


class FeatureTests(unittest.TestCase):
    def test_feature_shape_and_castling_rights(self):
        with_rights = features.board_to_features(chess.STARTING_FEN).reshape(8, 8, features.PLANES)
        without_rights = features.board_to_features(chess.STARTING_FEN.replace("KQkq", "-")).reshape(
            8, 8, features.PLANES
        )
        self.assertEqual(with_rights.shape, (8, 8, 18))
        self.assertTrue(np.all(with_rights[:, :, 13:17] == 1))
        self.assertTrue(np.all(without_rights[:, :, 13:17] == 0))

    def test_en_passant_square_is_encoded(self):
        fen = "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2"
        encoded = features.board_to_features(fen).reshape(8, 8, features.PLANES)
        row = 7 - chess.square_rank(chess.F3)
        col = chess.square_file(chess.F3)
        self.assertEqual(encoded[row, col, 17], 1.0)
        self.assertEqual(float(np.sum(encoded[:, :, 17])), 1.0)


class SearchAndModelTests(unittest.TestCase):
    def test_terminal_value_uses_side_to_move_perspective(self):
        checkmated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(checkmated.is_checkmate())
        self.assertEqual(self_play.terminal_value(checkmated), -1.0)

    def test_model_shapes_match_browser_contract(self):
        model = train.build_model()
        self.assertEqual(tuple(model.input_shape[1:]), (8, 8, features.PLANES))
        self.assertEqual(int(model.outputs[0].shape[-1]), policy_map.POLICY_SIZE)
        self.assertEqual(int(model.outputs[1].shape[-1]), 1)
        self.assertTrue(train.is_dual_head_model(model))

    def test_candidate_gate_fails_closed(self):
        self.assertEqual(
            train.should_accept_candidate(None, {"loss": 1.0}, resumed=True),
            (False, "candidate_validation_failed"),
        )
        self.assertEqual(
            train.should_accept_candidate({"loss": 0.9}, None, resumed=True),
            (False, "previous_validation_failed"),
        )


if __name__ == "__main__":
    unittest.main()
