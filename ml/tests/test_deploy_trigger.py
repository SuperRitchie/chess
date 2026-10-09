import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class DeployTriggerTests(unittest.TestCase):
    def test_successful_nightly_runs_trigger_master_deploy(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("workflow_run:", workflow)
        self.assertIn("Nightly NN training", workflow)
        self.assertIn("Nightly policy-value self training", workflow)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", workflow)
        self.assertIn("ref: master", workflow)


if __name__ == "__main__":
    unittest.main()
