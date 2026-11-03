"""Documentation and public-API consistency guards."""

import inspect
import os
import unittest

from inference import transcribe_video


class TestDocumentedApi(unittest.TestCase):
    def test_inference_signature_uses_stage1_model_path(self):
        params = inspect.signature(transcribe_video).parameters
        self.assertIn("stage1_model_path", params)
        self.assertNotIn("stage1_weights_path", params)

    def test_readme_examples_match_the_api(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as handle:
            readme = handle.read()
        self.assertNotIn("stage1_weights_path", readme)
        self.assertIn("stage1_model_path", readme)


if __name__ == "__main__":
    unittest.main()
