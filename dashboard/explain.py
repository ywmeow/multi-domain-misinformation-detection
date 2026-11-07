"""On-demand LIME word explanations for the dashboard's models.

Reuses ``lime_explain`` from ``explainability/explainers.py`` so the app and the
project's explainability analysis compute explanations the same way. LIME deletes
random subsets of words, re-scores each variant with the model and fits a local linear
model: a word's weight is how much it raises (+, towards fake) or lowers (-, towards
real) the fake-class score.

LIME needs hundreds of model calls per text, so it runs only when the user asks and
uses a smaller budget for the transformer, which is the slowest model on CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from time import perf_counter

from dashboard.inference import INFERENCE_LOCK, LOGGER, ROOT, ModelUnavailable, _module, load_detector

# (perturbed samples, words explained). More samples = more stable weights, slower.
# RoBERTa costs ~0.1 s per perturbed text on a 12-thread laptop CPU, so it gets less.
SETTINGS = {
    "tfidf": (500, 256),
    "bilstm": (300, 200),
    "graphsage": (300, 200),
    "transformer": (100, 100),
}
TOP_WORDS = 6        # words listed per direction
SENT_WEIGHTS = 60    # strongest words sent to the browser for highlighting
SEED = 0
# Below this, deleting any single word moves the fake-class score by < 1 point.
WEAK_EVIDENCE = 0.01

_explainers = None


def _lime():
    global _explainers
    if _explainers is None:
        _explainers = _module("dashboard_explainers", ROOT / "explainability/explainers.py")
    return _explainers


class _BlackBox:
    """The interface ``lime_explain`` expects: predict_proba and p_fake on text lists."""

    def __init__(self, name, detector):
        import numpy as np

        self.np = np
        self.name = name
        self.detector = detector

    def p_fake(self, texts):
        return self.detector.predict_batch(list(texts))

    def predict_proba(self, texts):
        p = self.p_fake(texts)
        return self.np.stack([1 - p, p], axis=1)


@contextmanager
def _all_cpu_threads(enabled: bool):
    """Let the slow transformer use every core while it is being explained.

    The app normally caps PyTorch at 4 threads (see BiLSTMDetector); this runs under
    INFERENCE_LOCK, so restoring the previous setting afterwards is safe.
    """
    if not enabled:
        yield
        return
    import torch
    from threadpoolctl import threadpool_limits

    previous = torch.get_num_threads()
    cores = os.cpu_count() or previous
    torch.set_num_threads(cores)
    try:
        with threadpool_limits(limits=cores):
            yield
    finally:
        torch.set_num_threads(previous)


def _explain_one(model: str, text: str) -> dict:
    samples, max_words = SETTINGS[model]
    try:
        detector = load_detector(model)
        started = perf_counter()
        with _all_cpu_threads(model == "transformer"):
            result = _lime().lime_explain(
                _BlackBox(model, detector), text, num_samples=samples, seed=SEED, max_words=max_words
            )
        ranked = sorted(result["weights"].items(), key=lambda kv: -abs(kv[1]))
        fake = [[w, round(v, 5)] for w, v in ranked if v > 0][:TOP_WORDS]
        real = [[w, round(v, 5)] for w, v in ranked if v < 0][:TOP_WORDS]
        return {
            "model": model,
            "status": "ready",
            "text": result["text"],  # the (possibly shortened) text that was explained
            "shortened": result["text"] != text,
            "max_words": max_words,
            "samples": samples,
            "weights": {w: round(v, 5) for w, v in ranked[:SENT_WEIGHTS]},
            "top_fake": fake,
            "top_real": real,
            # A very confident model barely reacts to deleting words: say so instead of
            # letting near-zero weights look meaningful.
            "weak_evidence": not ranked or abs(ranked[0][1]) < WEAK_EVIDENCE,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        }
    except ModelUnavailable as error:
        return {"model": model, "status": "unavailable", "message": str(error)}
    except Exception:
        LOGGER.exception("Explanation failed for %s", model)
        return {"model": model, "status": "error",
                "message": "This explanation could not run. Check the app's terminal."}


def explain(text: str, predictions: list[dict]) -> dict:
    """Explain every model that produced a prediction for ``text``."""
    from threadpoolctl import threadpool_limits

    started = perf_counter()
    ready = [p["model"] for p in predictions if p.get("status") == "ready"]
    with INFERENCE_LOCK, threadpool_limits(limits=4):
        results = [_explain_one(model, text) for model in ready if model in SETTINGS]
    return {
        "text": text,
        "method": "LIME",
        "explanations": results,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
    }
