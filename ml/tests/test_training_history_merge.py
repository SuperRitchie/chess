import importlib.util
import pathlib
import unittest


path = pathlib.Path(__file__).resolve().parents[2] / "scripts/merge_training_history.py"
spec = importlib.util.spec_from_file_location("merge_training_history", path)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


class TrainingHistoryMergeTests(unittest.TestCase):
    def test_preserves_both_jobs_when_the_second_started_from_older_history(self):
        old = {"timestamp_utc": "2026-10-08T12:00:00", "run_kind": "stockfish"}
        first = {"timestamp_utc": "2026-10-09T12:00:00", "run_kind": "stockfish"}
        second = {"timestamp_utc": "2026-10-09T13:00:00", "run_kind": "self_play"}
        self.assertEqual(merge.merge_records([old, first], [old, second]), [old, first, second])

    def test_repeated_staging_is_idempotent_and_retention_keeps_the_newest(self):
        records = [{"timestamp_utc": str(index).zfill(3)} for index in range(400)]
        result = merge.merge_records(records, records)
        self.assertEqual(len(result), 365)
        self.assertEqual(result[0], records[35])
        self.assertEqual(result[-1], records[-1])


if __name__ == "__main__":
    unittest.main()
