"""Guard evidence retention around the nightly runner-loss failure boundary."""
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = {
    "nightly-train.yml": (
        "Preserve generated Stockfish labels", "ml/data/labels.json",
        "Continue NN learning, evaluate, and export TFJS",
        "Commit updated brain and browser model",
    ),
    "nightly-self-play.yml": (
        "Preserve generated self-play replay", "ml/data/self_play_buffer.json",
        "Continue policy-value learning, evaluate, and export model",
        "Commit updated model state",
    ),
}


class NightlyEvidenceTests(unittest.TestCase):
    def test_completed_generation_is_archived_before_training(self):
        for filename, (preserve, path, train, _) in WORKFLOWS.items():
            with self.subTest(workflow=filename):
                text = (ROOT / ".github/workflows" / filename).read_text()
                archive = text.split(f"- name: {preserve}", 1)[1].split(f"- name: {train}", 1)[0]
                self.assertIn("uses: actions/upload-artifact@v7", archive)
                self.assertIn(path, archive)
                self.assertIn("if-no-files-found: error", archive)
                self.assertIn("retention-days: 14", archive)
                self.assertIn("${{ github.run_id }}-${{ github.run_attempt }}", archive)
                # Default success gating must not archive stale inputs after failed generation.
                self.assertNotIn("always()", archive)

    def test_training_failure_remains_a_failure_and_keeps_evidence(self):
        for filename, (_, _, train, commit) in WORKFLOWS.items():
            with self.subTest(workflow=filename):
                text = (ROOT / ".github/workflows" / filename).read_text()
                block = text.split(f"- name: {train}", 1)[1].split(f"- name: {commit}", 1)[0]
                self.assertIn("id: train", block)
                self.assertIn("set -o pipefail", block)
                self.assertIn('python ml/train_safe_export.py 2>&1 | tee "$RUNNER_TEMP/nightly-training.log"', block)
                self.assertIn("if: ${{ always() && steps.train.outcome != 'skipped' }}", block)
                self.assertIn("${{ runner.temp }}/nightly-training.log", block)
                self.assertIn("ml/training_history.json", block)
                self.assertIn("public/nn/group*.bin", block)
                self.assertIn("retention-days: 14", block)
                self.assertIn("${{ github.run_id }}-${{ github.run_attempt }}", block)


if __name__ == "__main__":
    unittest.main()
