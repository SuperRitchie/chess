import concurrent.futures
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chess.engine
import stockfish_eval


class StockfishRecoveryTests(unittest.TestCase):
    def test_crash_or_timeout_restarts_and_retries_the_same_position(self):
        for error in (chess.engine.EngineTerminatedError("exit -11"), concurrent.futures.TimeoutError()):
            with self.subTest(error=type(error).__name__):
                dead, healthy = mock.Mock(), mock.Mock()
                cache = {}
                with (
                    mock.patch.object(stockfish_eval.chess.engine.SimpleEngine, "popen_uci", side_effect=[dead, healthy]) as start,
                    mock.patch.object(stockfish_eval, "analyse_position", side_effect=[{"fen": "first"}, error, {"fen": "second"}, {"fen": "third"}]) as analyse,
                ):
                    stockfish_eval.evaluate_missing_positions(["first", "second", "third"], cache)
                self.assertEqual(start.call_count, 2)
                self.assertEqual([c.args[1] for c in analyse.call_args_list], ["first", "second", "second", "third"])
                self.assertIs(analyse.call_args_list[2].args[0], healthy)
                self.assertEqual(set(cache), {"first", "second", "third"})
                dead.quit.assert_not_called()
                dead.close.assert_called_once()
                healthy.quit.assert_called_once()
                healthy.close.assert_called_once()

    def test_persistent_crashes_fail_closed_without_replacing_existing_labels(self):
        engines = [mock.Mock(), mock.Mock(), mock.Mock()]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "labels.json"
            original = '[{"fen":"previous","policy":[[1,1.0]]}]'
            output.write_text(original, encoding="utf-8")
            with (
                mock.patch.object(stockfish_eval, "OUT_JSON", output),
                mock.patch.object(stockfish_eval, "MAX_ENGINE_RESTARTS", 2),
                mock.patch.object(stockfish_eval, "read_unique_fens", return_value=["first", "poison", "third"]),
                mock.patch.object(stockfish_eval, "read_cached_labels", return_value={}),
                mock.patch.object(stockfish_eval.chess.engine.SimpleEngine, "popen_uci", side_effect=engines) as start,
                mock.patch.object(stockfish_eval, "analyse_position", side_effect=[{"fen": "first"}] + [chess.engine.EngineTerminatedError("exit -11")] * 3) as analyse,
            ):
                with self.assertRaisesRegex(RuntimeError, "restart budget exhausted"):
                    stockfish_eval.main()
            self.assertEqual(output.read_text(encoding="utf-8"), original)
        self.assertEqual(start.call_count, 3)
        self.assertEqual([c.args[1] for c in analyse.call_args_list], ["first", "poison", "poison", "poison"])
        for engine in engines:
            engine.close.assert_called_once()
            engine.quit.assert_not_called()

    def test_no_pv_is_skipped_without_restarting_a_healthy_engine(self):
        engine = mock.Mock()
        cache = {}
        with (
            mock.patch.object(stockfish_eval.chess.engine.SimpleEngine, "popen_uci", return_value=engine) as start,
            mock.patch.object(stockfish_eval, "analyse_position", side_effect=[ValueError("no principal variation"), {"fen": "second"}]),
        ):
            stockfish_eval.evaluate_missing_positions(["terminal", "second"], cache)
        start.assert_called_once()
        self.assertEqual(set(cache), {"second"})
        engine.quit.assert_called_once()
        engine.close.assert_called_once()

    def test_cleanup_of_an_already_dead_engine_is_safe(self):
        for error in (chess.engine.EngineTerminatedError("event loop dead"), concurrent.futures.TimeoutError()):
            with self.subTest(error=type(error).__name__):
                engine = mock.Mock()
                engine.quit.side_effect = error
                stockfish_eval.close_engine(engine)
                engine.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
