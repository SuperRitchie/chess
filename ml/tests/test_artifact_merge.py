import pathlib
import sys
import unittest


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from merge_training_history import merge_records


class ArtifactMergeTests(unittest.TestCase):
    def test_concurrent_run_records_are_preserved_once(self):
        base = {"timestamp_utc": "2026-08-23T10:00:00+00:00", "run_kind": "stockfish"}
        current = [base, {"timestamp_utc": "2026-08-24T10:00:00+00:00", "run_kind": "stockfish"}]
        artifact = [base, {"timestamp_utc": "2026-08-24T12:00:00+00:00", "run_kind": "self_play"}]

        merged = merge_records(current, artifact)

        self.assertEqual(len(merged), 3)
        self.assertEqual([item["run_kind"] for item in merged], ["stockfish", "stockfish", "self_play"])

    def test_duplicate_timestamp_uses_artifact_record(self):
        timestamp = "2026-08-24T12:00:00+00:00"
        merged = merge_records(
            [{"timestamp_utc": timestamp, "candidate_accepted": False}],
            [{"timestamp_utc": timestamp, "candidate_accepted": True}],
        )

        self.assertTrue(merged[0]["candidate_accepted"])


if __name__ == "__main__":
    unittest.main()
