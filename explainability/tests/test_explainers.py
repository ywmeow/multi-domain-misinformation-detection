"""Quick checks with a toy model whose behaviour is known exactly.

Run: python -m pytest explainability/tests  (or python explainability/tests/test_explainers.py)
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explainers import deletion_check, highlight_html, lime_explain, shap_explain, truncate_words  # noqa: E402
from models import Model  # noqa: E402


class KeywordModel(Model):
    """P(fake) rises with 'shocking' and falls with 'reuters'; other words do nothing."""

    name = "toy"

    def predict_proba(self, texts):
        p = []
        for text in texts:
            words = text.lower().split()
            score = 1.5 * words.count("shocking") - 1.5 * words.count("reuters")
            p.append(1 / (1 + np.exp(-score)))
        p = np.asarray(p)
        return np.stack([1 - p, p], axis=1)


TEXT = "shocking claim spreads online while reuters checks the shocking story"


def test_lime_finds_known_cues():
    weights = lime_explain(KeywordModel(), TEXT, num_samples=400)["weights"]
    assert weights["shocking"] > 0.05
    assert weights["reuters"] < -0.05
    assert abs(weights["online"]) < abs(weights["shocking"])


def test_shap_finds_known_cues():
    result = shap_explain(KeywordModel(), TEXT, max_evals=200)
    weights = result["weights"]
    assert weights["shocking"] > 0 and weights["reuters"] < 0
    # Shapley values add up to the prediction minus the fully-masked baseline.
    assert abs(sum(result["token_values"]) - (result["p_fake"] - result["base_value"])) < 1e-6


def test_deletion_check_prefers_explained_words():
    model = KeywordModel()
    weights = lime_explain(model, TEXT, num_samples=400)["weights"]
    check = deletion_check(model, TEXT, weights, k=1)
    assert check["explained_drop"] > check["random_drop"]


def test_truncate_and_highlight():
    assert truncate_words("a b c d", 2) == "a b"
    page = highlight_html("shocking <> news", {"shocking": 0.3})
    assert "&lt;&gt;" in page and "230,120,30" in page


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
