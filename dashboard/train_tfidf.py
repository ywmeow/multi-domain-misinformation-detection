"""Train and save the TF-IDF + logistic regression baseline used by the dashboard.

Run once from the repository root (about a minute on CPU):

    python dashboard/train_tfidf.py

The settings are those of ``trung/baseline_tfidf.py`` (word 1-2 grams, min_df 3,
up to 500k features, sublinear tf; logistic regression C=10 with balanced class
weights), fitted on the pooled training splits of all four datasets like the other
dashboard models. The result is written to ``dashboard/models/tfidf_lr.joblib`` together
with its settings and test scores, so the app never trains anything itself.
"""

from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "explainability"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dashboard.inference import POOLED_SOURCES, TFIDF_CHECKPOINT  # noqa: E402

VECTORIZER = dict(ngram_range=(1, 2), min_df=3, max_features=500_000, sublinear_tf=True)
CLASSIFIER = dict(C=10, max_iter=2000, class_weight="balanced")


def main() -> None:
    import joblib
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline

    # Shared loader for the team's binary exports (extracts the 7z archives if needed).
    from data import load_pooled

    train, test = load_pooled("train", POOLED_SOURCES), load_pooled("test", POOLED_SOURCES)
    pipeline = make_pipeline(TfidfVectorizer(**VECTORIZER), LogisticRegression(**CLASSIFIER))
    started = time.time()
    pipeline.fit(train.text, train.label_id)
    seconds = time.time() - started

    predicted = pipeline.predict(test.text)
    scores = {"overall": f1_score(test.label_id, predicted, average="macro")}
    for dataset, group in test.assign(predicted=predicted).groupby("dataset"):
        scores[dataset] = f1_score(group.label_id, group.predicted, average="macro")

    bundle = {
        "architecture": "tfidf + logistic regression",
        "pipeline": pipeline,
        "training_sources": list(POOLED_SOURCES),
        "threshold": 0.5,
        "settings": {"vectorizer": VECTORIZER, "classifier": CLASSIFIER},
        "train_rows": len(train),
        "features": len(pipeline[0].vocabulary_),
        "training_seconds": round(seconds, 1),
        "test_macro_f1": {k: round(float(v), 4) for k, v in scores.items()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    TFIDF_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, TFIDF_CHECKPOINT, compress=3)
    size = TFIDF_CHECKPOINT.stat().st_size / 1e6
    print(f"Saved {TFIDF_CHECKPOINT.relative_to(ROOT)} ({size:.1f} MB, {bundle['features']:,} features, "
          f"trained on {len(train):,} texts in {seconds:.0f} s)")
    print("Test macro-F1:", bundle["test_macro_f1"])


if __name__ == "__main__":
    main()
