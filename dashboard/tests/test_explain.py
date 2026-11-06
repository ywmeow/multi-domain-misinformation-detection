"""Checks for the on-demand LIME explanations, using toy models with known cues."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dashboard import explain, inference


def keyword_detector():
    """P(fake) rises with 'shocking' and falls with 'reuters'; other words do nothing."""

    def predict_batch(texts):
        words = [text.lower().split() for text in texts]
        score = np.array([1.5 * w.count("shocking") - 1.5 * w.count("reuters") for w in words])
        return 1 / (1 + np.exp(-score))

    return SimpleNamespace(predict_batch=predict_batch)


TEXT = "shocking claim spreads online while reuters checks the shocking story"
READY = [{"model": "bilstm", "status": "ready"}]


class ExplanationTests(unittest.TestCase):
    def test_finds_the_known_cue_words(self):
        with patch.object(explain, "load_detector", return_value=keyword_detector()):
            result = explain.explain(TEXT, READY)
        item = result["explanations"][0]
        self.assertEqual(item["status"], "ready")
        self.assertEqual(item["top_fake"][0][0], "shocking")
        self.assertEqual(item["top_real"][0][0], "reuters")
        self.assertGreater(item["weights"]["shocking"], 0)
        self.assertLess(item["weights"]["reuters"], 0)
        self.assertFalse(item["shortened"])

    def test_explains_only_models_that_predicted(self):
        predictions = [
            {"model": "bilstm", "status": "ready"},
            {"model": "graphsage", "status": "error"},
            {"model": "transformer", "status": "unavailable"},
        ]
        with patch.object(explain, "load_detector", return_value=keyword_detector()) as loader:
            result = explain.explain(TEXT, predictions)
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["bilstm"])
        self.assertEqual([item["model"] for item in result["explanations"]], ["bilstm"])

    def test_long_text_is_shortened_to_the_word_budget(self):
        _, max_words = explain.SETTINGS["bilstm"]
        long_text = " ".join(["shocking", "story"] * max_words)
        with patch.object(explain, "load_detector", return_value=keyword_detector()):
            item = explain.explain(long_text, READY)["explanations"][0]
        self.assertTrue(item["shortened"])
        self.assertEqual(len(item["text"].split()), max_words)

    def test_failure_is_isolated(self):
        broken = SimpleNamespace(predict_batch=lambda texts: 1 / 0)
        with patch.object(explain, "load_detector", return_value=broken):
            with self.assertLogs(inference.__name__, level="ERROR"):
                item = explain.explain(TEXT, READY)["explanations"][0]
        self.assertEqual(item["status"], "error")


class RealModelExplanationTests(unittest.TestCase):
    def test_bilstm_and_graphsage_explanations_run(self):
        predictions = [{"model": "bilstm", "status": "ready"}, {"model": "graphsage", "status": "ready"}]
        result = explain.explain("Officials said on Tuesday that the new bridge will open in May.", predictions)
        for item in result["explanations"]:
            with self.subTest(model=item["model"]):
                self.assertEqual(item["status"], "ready")
                self.assertTrue(item["weights"])


if __name__ == "__main__":
    unittest.main()
