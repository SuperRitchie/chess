import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import chess
import chess.engine
import numpy as np

ML_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import features
import fen_utils
import policy_map
import self_play
import stockfish_eval
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


class DataPipelineTests(unittest.TestCase):
    def test_canonical_fen_removes_move_counters(self):
        first = "8/8/8/8/8/8/4K3/7k w - - 17 48"
        second = "8/8/8/8/8/8/4K3/7k w - - 0 1"
        self.assertEqual(fen_utils.canonical_fen(first), fen_utils.canonical_fen(second))

    def test_stockfish_analysis_creates_policy_target(self):
        class FakeEngine:
            def analyse(self, board, limit, multipv):
                return [
                    {
                        "score": chess.engine.PovScore(chess.engine.Cp(score), chess.WHITE),
                        "pv": [chess.Move.from_uci(move)],
                    }
                    for move, score in (("e2e4", 80), ("d2d4", 50), ("g1f3", 20))
                ]

        label = stockfish_eval.analyse_position(FakeEngine(), chess.STARTING_FEN)

        self.assertEqual(label["policy_version"], policy_map.POLICY_VERSION)
        self.assertEqual(len(label["policy"]), 3)
        self.assertAlmostEqual(sum(item[1] for item in label["policy"]), 1.0)
        self.assertEqual(label["policy"][0][0], policy_map.move_to_index(chess.Move.from_uci("e2e4")))

    def test_replay_freshness_counts_only_unseen_positions(self):
        first_fen = fen_utils.canonical_fen(chess.STARTING_FEN)
        board = chess.Board()
        board.push_uci("e2e4")
        second_fen = fen_utils.canonical_fen(board.fen(en_passant="fen"))
        first_policy = [[policy_map.move_to_index(chess.Move.from_uci("e2e4")), 1.0]]
        with tempfile.TemporaryDirectory() as directory:
            replay_path = pathlib.Path(directory) / "replay.json"
            replay_path.write_text(
                json.dumps(
                    [
                        {
                            "fen": first_fen,
                            "cp": 0,
                            "policy_version": policy_map.POLICY_VERSION,
                            "policy": first_policy,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(train, "STOCKFISH_REPLAY_BUFFER", replay_path):
                merged, novel_count = train.merge_stockfish_replay_buffer(
                    [{"fen": first_fen, "cp": 10}, {"fen": second_fen, "cp": 20}]
                )

        self.assertEqual(novel_count, 1)
        self.assertEqual({item["fen"] for item in merged}, {first_fen, second_fen})
        self.assertEqual(next(item for item in merged if item["fen"] == first_fen)["policy"], first_policy)


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

    def test_batched_search_shares_model_calls_across_games(self):
        class CountingModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, features_batch, training=False):
                self.calls += 1
                batch_size = len(features_batch)
                return [
                    np.zeros((batch_size, policy_map.POLICY_SIZE), dtype=np.float32),
                    np.zeros((batch_size, 1), dtype=np.float32),
                ]

        model = CountingModel()
        boards = [chess.Board(), chess.Board()]
        policies = self_play.run_search_batch(model, boards, searches=3, add_noise=False)

        self.assertEqual(model.calls, 4)
        self.assertEqual(len(policies), 2)
        for board, policy in zip(boards, policies):
            legal_indices = {policy_map.move_to_index(move) for move in board.legal_moves}
            self.assertEqual(set(policy), legal_indices)
            self.assertAlmostEqual(sum(policy.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
