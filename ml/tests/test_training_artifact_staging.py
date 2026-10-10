import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class TrainingArtifactStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)
        self.remote = self.directory / "remote.git"
        self.repo = self.directory / "runner"
        self.run_command("git", "init", "--bare", "--initial-branch=master", str(self.remote))
        self.run_command("git", "clone", str(self.remote), str(self.repo))
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.com")
        for relative, content in {
            "public/nn/model.json": "accepted browser model",
            "ml/checkpoints/model.keras": "accepted checkpoint",
            "ml/data/replay_buffer.json": "[]",
            "ml/data/fixed_eval_set_v3.json": "[]",
            "ml/training_history.json": '[]',
        }.items():
            self.write(relative, content)
        (self.repo / "scripts").mkdir()
        for name in ("stage_model_artifacts.sh", "merge_training_history.py"):
            shutil.copy2(ROOT / "scripts" / name, self.repo / "scripts" / name)
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.git("push", "origin", "master")

    def run_command(self, *command, cwd=None, check=True):
        return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)

    def git(self, *args):
        return self.run_command("git", *args, cwd=self.repo)

    def write(self, relative, content):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def stage(self, check=True):
        return self.run_command("bash", "scripts/stage_model_artifacts.sh", "ml/data/replay_buffer.json", cwd=self.repo, check=check)

    def test_rejected_candidate_cannot_publish_its_browser_or_checkpoint_files(self):
        self.write("public/nn/model.json", "rejected browser model")
        self.write("ml/checkpoints/model.keras", "rejected checkpoint")
        self.write("ml/training_history.json", json.dumps([{"timestamp_utc": "2026-10-09", "candidate_accepted": False}]))
        self.stage()
        self.assertEqual((self.repo / "public/nn/model.json").read_text(), "accepted browser model")
        self.assertEqual((self.repo / "ml/checkpoints/model.keras").read_text(), "accepted checkpoint")
        self.assertEqual(len(json.loads((self.repo / "ml/training_history.json").read_text())), 1)

    def test_accepted_candidate_publishes_its_browser_and_checkpoint_together(self):
        self.write("public/nn/model.json", "new accepted browser model")
        self.write("ml/checkpoints/model.keras", "new accepted checkpoint")
        self.write("ml/training_history.json", json.dumps([{"timestamp_utc": "2026-10-09", "candidate_accepted": True}]))
        self.stage()
        self.assertEqual((self.repo / "public/nn/model.json").read_text(), "new accepted browser model")
        self.assertEqual((self.repo / "ml/checkpoints/model.keras").read_text(), "new accepted checkpoint")

    def test_newer_remote_model_rejects_an_artifact_from_an_older_checkpoint(self):
        other = self.directory / "other"
        self.run_command("git", "clone", str(self.remote), str(other))
        self.run_command("git", "config", "user.name", "test", cwd=other)
        self.run_command("git", "config", "user.email", "test@example.com", cwd=other)
        (other / "public/nn/model.json").write_text("newer model")
        self.run_command("git", "add", ".", cwd=other)
        self.run_command("git", "commit", "-m", "new accepted weights", cwd=other)
        self.run_command("git", "push", "origin", "master", cwd=other)
        result = self.stage(check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("changed while this job was running", result.stdout)


if __name__ == "__main__":
    unittest.main()
