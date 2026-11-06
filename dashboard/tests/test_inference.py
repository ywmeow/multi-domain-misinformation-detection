"""Regression checks for saved preprocessing, thresholds, and partial failure."""

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dashboard import inference


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputAndFailureTests(unittest.TestCase):
    def test_normalizes_unicode_and_whitespace(self):
        self.assertEqual(inference.normalize_text("  Cafe\u0301\n  news\t "), "Café news")

    def test_rejects_blank_and_oversize_inputs(self):
        for text in ("", " \n\t", None, "x" * (inference.MAX_CHARACTERS + 1)):
            with self.subTest(text_type=type(text).__name__):
                with self.assertRaises(ValueError):
                    inference.normalize_text(text)

    def test_saved_threshold_and_absent_transformer(self):
        detector = SimpleNamespace(predict=lambda text: (0.53, 0.55, False))
        with patch.object(inference, "load_detector", return_value=detector) as loader:
            result = inference.analyze("A test claim.")
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["bilstm", "graphsage"])
        self.assertEqual(result["predictions"][1]["label"], "real")
        self.assertEqual(result["predictions"][2]["status"], "unavailable")
        self.assertIsNone(result["predictions"][2]["p_fake"])

    def test_failure_preserves_other_prediction_without_claiming_agreement(self):
        detector = SimpleNamespace(predict=lambda text: (0.7, 0.55, False))
        with patch.object(inference, "load_detector", side_effect=[FileNotFoundError(), detector]):
            with self.assertLogs(inference.__name__, level="ERROR"):
                result = inference.analyze("A test claim.")
        self.assertEqual(result["predictions"][0]["status"], "error")
        self.assertEqual(result["predictions"][1]["status"], "ready")
        self.assertEqual(result["completed"], 1)
        self.assertIsNone(result["agreement"])

    def test_invalid_probability_becomes_an_isolated_error(self):
        detector = SimpleNamespace(predict=lambda text: (float("nan"), 0.5, False))
        with patch.object(inference, "load_detector", return_value=detector):
            with self.assertLogs(inference.__name__, level="ERROR"):
                result = inference.analyze("A test claim.")
        self.assertEqual(result["completed"], 0)
        self.assertIsNone(result["agreement"])


class NativeCheckpointParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = inference.ROOT
        cls.bilstm = inference.load_detector("bilstm")
        cls.graphsage = inference.load_detector("graphsage")
        # Existing CLI modules use unqualified imports. Isolate these imports
        # only for the test, without changing application import resolution.
        sequence_model = import_file("native_sequence", root / "sequential/model.py")
        with patch.dict(sys.modules, {"model": sequence_model}):
            sequence_predict = import_file("native_predict", root / "sequential/predict.py")
        cls.native_bilstm = sequence_predict.MisinformationDetector()
        graph_model = import_file("native_graph", root / "graph/model.py")
        with patch.dict(sys.modules, {"model": graph_model, "prep_data": SimpleNamespace(BASE=root / "graph")}):
            graph_predict = import_file("native_graph_predict", root / "graph/implementation.py")
        cls.native_graph = graph_predict.MisinformationDetector(dataset="jruvika")

    def test_bilstm_matches_original_predictor(self):
        for text in ("The council approved a new public transport plan.", "Cafe\u0301\n  news in the city."):
            with self.subTest(text=text):
                score, threshold, _ = self.bilstm.predict(inference.normalize_text(text))
                expected = self.native_bilstm.predict(text)
                self.assertAlmostEqual(score, expected["p_fake"], places=6)
                self.assertEqual(int(score >= threshold), expected["label_id"])

    def test_graphsage_matches_original_predictor_and_threshold(self):
        text = "Researchers reported a new discovery after a laboratory study."
        score, threshold, _ = self.graphsage.predict(text)
        expected = self.native_graph.predict([text])[0]
        self.assertAlmostEqual(score, expected["fake_probability"], places=6)
        self.assertEqual(threshold, self.native_graph.bundle["threshold"])
        self.assertEqual(int(score >= threshold), expected["label_id"])

    def test_long_text_matches_native_truncation(self):
        text = "word " * 520
        score, _, truncated = self.bilstm.predict(text)
        self.assertTrue(truncated)
        self.assertAlmostEqual(score, self.native_bilstm.predict(text)["p_fake"], places=6)


if __name__ == "__main__":
    unittest.main()
