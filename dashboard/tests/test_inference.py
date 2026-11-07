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

    def test_saved_threshold_applies_to_every_model(self):
        detector = SimpleNamespace(predict=lambda text: (0.53, 0.55, False))
        with patch.object(inference, "load_detector", return_value=detector) as loader:
            result = inference.analyze("A test claim.")
        self.assertEqual([call.args[0] for call in loader.call_args_list], list(inference.MODELS))
        self.assertEqual([p["label"] for p in result["predictions"]], ["real"] * 4)
        self.assertEqual(result["agreement"], "agree")

    def test_missing_transformer_is_unavailable_not_an_error(self):
        detector = SimpleNamespace(predict=lambda text: (0.7, 0.5, False))
        missing = inference.ModelUnavailable("RoBERTa weights are not on this machine.")
        with patch.object(inference, "load_detector", side_effect=[detector, detector, detector, missing]):
            result = inference.analyze("A test claim.")
        transformer = result["predictions"][3]
        self.assertEqual(transformer["model"], "transformer")
        self.assertEqual(transformer["status"], "unavailable")
        self.assertIsNone(transformer["p_fake"])
        self.assertIn("not on this machine", transformer["message"])
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["agreement"], "agree")

    def test_disagreement_counts_fake_votes(self):
        scores = iter([0.9, 0.2, 0.8, 0.1])
        detector = SimpleNamespace(predict=lambda text: (next(scores), 0.5, False))
        with patch.object(inference, "load_detector", return_value=detector):
            result = inference.analyze("A test claim.")
        self.assertEqual(result["agreement"], "disagree")
        self.assertEqual(result["fake_votes"], 2)

    def test_failure_preserves_other_prediction_without_claiming_agreement(self):
        detector = SimpleNamespace(predict=lambda text: (0.7, 0.55, False))
        missing = inference.ModelUnavailable("not set up")
        failures = [FileNotFoundError(), detector, FileNotFoundError(), missing]
        with patch.object(inference, "load_detector", side_effect=failures):
            with self.assertLogs(inference.__name__, level="ERROR"):
                result = inference.analyze("A test claim.")
        self.assertEqual(result["predictions"][0]["status"], "error")
        self.assertEqual(result["predictions"][1]["status"], "ready")
        self.assertEqual(result["predictions"][2]["status"], "error")
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
        cls.tfidf = inference.load_detector("tfidf")
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
        # The pooled checkpoint, loaded the way graph/Model_pooled/predict.py does.
        cls.native_graph = graph_predict.MisinformationDetector(path=inference.GRAPHSAGE_CHECKPOINT)

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

    def test_batch_scores_match_single_predictions(self):
        texts = ["The council approved a new public transport plan.",
                 "Shocking video shows what they are hiding from you."]
        detectors = (("tfidf", self.tfidf), ("bilstm", self.bilstm), ("graphsage", self.graphsage))
        for name, detector in detectors:
            with self.subTest(model=name):
                batch = detector.predict_batch(texts)
                for text, score in zip(texts, batch):
                    self.assertAlmostEqual(float(score), detector.predict(text)[0], places=5)

    def test_graphsage_uses_the_pooled_checkpoint(self):
        bundle = self.graphsage.bundle
        self.assertEqual(sorted(bundle["training_sources"]), sorted(inference.POOLED_SOURCES))
        self.assertEqual(self.graphsage.threshold, bundle["threshold"])

    def test_tfidf_matches_saved_pipeline_and_baseline_score(self):
        text = "Officials said on Tuesday that the council approved the plan."
        score, threshold, truncated = self.tfidf.predict(text)
        expected = self.tfidf.pipeline.predict_proba([text])[0, 1]
        self.assertAlmostEqual(score, float(expected), places=6)
        self.assertEqual(threshold, 0.5)
        self.assertFalse(truncated)
        # trung/results/tfidf_lr/metrics.json records 0.8144 for the same settings.
        self.assertAlmostEqual(self.tfidf.bundle["test_macro_f1"]["overall"], 0.8144, delta=0.002)

    def test_long_text_matches_native_truncation(self):
        text = "word " * 520
        score, _, truncated = self.bilstm.predict(text)
        self.assertTrue(truncated)
        self.assertAlmostEqual(score, self.native_bilstm.predict(text)["p_fake"], places=6)


class TransformerCheckpointTests(unittest.TestCase):
    """Runs only once the RoBERTa weights have been downloaded."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.detector = inference.load_detector("transformer")
        except inference.ModelUnavailable as error:
            raise unittest.SkipTest(str(error))

    def test_matches_plain_transformers_usage(self):
        # Same steps as the usage example in Trung's Hugging Face README.
        import torch
        text = "The council approved a new public transport plan after a public consultation."
        encoded = self.detector.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            expected = self.detector.model(**encoded).logits.softmax(-1)[0, 1].item()
        score, threshold, truncated = self.detector.predict(text)
        self.assertAlmostEqual(score, expected, places=5)
        self.assertEqual(threshold, 0.5)
        self.assertFalse(truncated)

    def test_reports_truncation_beyond_512_tokens(self):
        self.assertTrue(self.detector.predict("word " * 600)[2])


if __name__ == "__main__":
    unittest.main()
