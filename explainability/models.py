"""One interface for every team model: ``predict_proba(list[str]) -> (n, 2)``.

Column 0 is P(real) and column 1 is P(fake). LIME and SHAP only need this black-box
function, so the same explainer code runs on every model family:

* ``tfidf_lr``           traditional baseline (TF-IDF 1-2 gram + logistic regression,
                         same settings as ``trung/baseline_tfidf.py``), pooled training.
* ``tfidf_lr:<dataset>`` the same baseline trained on one dataset only (cross-domain).
* ``bilstm``             sequential model, the shipped GloVe BiLSTM checkpoint (pooled).
* ``graphsage:<dataset>`` graph model, one of Swetha's four single-source checkpoints.
* ``graphsage:pooled``   her GraphSAGE trained on all four datasets (``Model_pooled.joblib``).
* ``transformer:<run>``  optional: a fine-tuned RoBERTa/DeBERTa from Hugging Face, by
                         Trung's run name (``roberta_adamw``, ``roberta_muon_lr5e-5``,
                         ``deberta_adamw``); needs ``transformers`` and a download.

Models are loaded from the existing team folders without changing them. Modules are
imported by file path because ``sequential/`` and ``graph/`` both have a ``model.py``.
"""

from functools import lru_cache
import importlib.util
import json
import time
import unicodedata

import joblib
import numpy as np

from data import CACHE, DATASETS, ROOT, load_pooled

TFIDF_DIR = CACHE / "models"


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    name = "model"
    family = "-"
    threshold = 0.5  # P(fake) at or above this is labelled fake

    def predict(self, texts):
        return (self.p_fake(texts) >= self.threshold).astype(int)

    def predict_proba(self, texts):
        raise NotImplementedError

    def p_fake(self, texts):
        return self.predict_proba(texts)[:, 1]


class TfidfLR(Model):
    """TF-IDF + logistic regression, fitted on the training split and cached."""

    family = "traditional"

    def __init__(self, train_sets=DATASETS):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.train_sets = tuple(train_sets)
        pooled = self.train_sets == DATASETS
        self.name = "tfidf_lr" if pooled else f"tfidf_lr:{self.train_sets[0]}"
        path = TFIDF_DIR / f"{self.name.replace(':', '_')}.joblib"
        if path.exists():
            self.pipeline = joblib.load(path)
            return
        train = load_pooled("train", self.train_sets)
        # Settings follow trung/baseline_tfidf.py; min_df relaxed for the tiny AMT set.
        min_df = 3 if len(train) > 2000 else 1
        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=500_000, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
        )
        start = time.time()
        self.pipeline.fit(train.text, train.label_id)
        print(f"[{self.name}] trained on {len(train)} texts in {time.time() - start:.0f}s")
        TFIDF_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path, compress=3)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(list(texts))

    def coefficients(self):
        """Exact global weights (positive = pushes towards fake); used to sanity-check LIME."""
        vec, clf = self.pipeline[0], self.pipeline[-1]
        return dict(zip(vec.get_feature_names_out(), clf.coef_[0]))


class BiLSTMModel(Model):
    """Batched version of ``sequential/predict.py`` (same preprocessing and vocabulary)."""

    name = "bilstm"
    family = "sequential"

    def __init__(self, checkpoint_dir=ROOT / "sequential/checkpoints/bilstm_glove100_seed42", batch_size=64):
        import spacy
        import torch

        seq = _import_file("seq_model", ROOT / "sequential/model.py")
        self.torch = torch
        self.batch_size = batch_size
        self.vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
        settings = json.loads((checkpoint_dir / "tokenizer_settings.json").read_text(encoding="utf-8"))
        self.max_length = settings["max_length"]
        self.unk = self.vocab["<UNK>"]
        self.pad = self.vocab["<PAD>"]
        self.tokenizer = spacy.blank("en")
        checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
        self.model = seq.BiLSTM(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def _ids(self, text):
        text = " ".join(unicodedata.normalize("NFC", text).split())
        ids = [self.vocab.get(t.text, self.unk) for t in self.tokenizer(text)[: self.max_length]]
        return ids or [self.unk]

    def predict_proba(self, texts):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [self._ids(t) for t in texts[start : start + self.batch_size]]
                lengths = torch.tensor([len(b) for b in batch])
                padded = torch.full((len(batch), int(lengths.max())), self.pad, dtype=torch.long)
                for row, ids in enumerate(batch):
                    padded[row, : len(ids)] = torch.tensor(ids)
                out.append(self.model(padded, lengths).softmax(dim=1).numpy())
        return np.concatenate(out)


class GraphSAGEModel(Model):
    """Inference for Swetha's saved GraphSAGE models (numpy only).

    ``dataset`` is one of the four datasets (a single-source model) or ``"pooled"``
    (``Model_pooled.joblib``, trained on all four). A new text becomes a node: TF-IDF ->
    SVD features, connected to its most similar training documents (the checkpoint's
    saved neighbour count), then passed through the two GraphSAGE layers.
    """

    family = "graph"

    def __init__(self, dataset="multifc", batch_size=256):
        from sklearn.preprocessing import normalize

        self.sage = _import_file("swetha_model", ROOT / "graph/model.py")
        self.normalize = normalize
        self.dataset = dataset
        self.name = f"graphsage:{dataset}"
        self.batch_size = batch_size
        filename = "Model_pooled.joblib" if dataset == "pooled" else f"{dataset}_graphsage.joblib"
        self.bundle = joblib.load(ROOT / "graph/Models" / filename)
        # The retrained checkpoints carry a decision threshold selected on source dev.
        self.threshold = float(self.bundle.get("threshold", 0.5))

    def predict_proba(self, texts):
        b = self.bundle
        out = []
        for start in range(0, len(texts), self.batch_size):
            x = self.normalize(b["svd"].transform(b["vectorizer"].transform(texts[start : start + self.batch_size])))
            x = x.astype("float32")
            adjacency = self.sage.query_adjacency(x, b["neighbors"], len(b["anchors"]), b["k"])
            logits = self.sage.numpy_logits(x, adjacency, b["anchors"], b["anchor_hidden"], b["state"])
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(exp / exp.sum(axis=1, keepdims=True))
        return np.concatenate(out)


TRANSFORMER_REPO = "ngqtrung/ee6405-misinfo-detection"
# Same commit as the dashboard (dashboard/inference.py), so both load identical weights.
TRANSFORMER_REVISION = "7837df88331f0b16f9bcbda8fc5e0e4f859a8ccc"
# Trung's run names (trung/results/<run>) and the Hugging Face folders that hold them.
TRANSFORMER_RUNS = {
    "roberta_adamw": "models/roberta-base-adamw",
    "roberta_muon_lr5e-5": "models/roberta-base-muon",
    "deberta_adamw": "models/deberta-v3-base-adamw",
}


class TransformerModel(Model):
    """Optional: one of Trung's fine-tuned checkpoints from Hugging Face.

    ``run`` is a run name from ``TRANSFORMER_RUNS``. Weights already in the local
    Hugging Face cache (e.g. fetched by ``dashboard/download_transformer.py``) are reused;
    otherwise ``transformers`` downloads them (about 0.5 GB each).
    """

    family = "transformer"

    def __init__(self, run="roberta_adamw", batch_size=16, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if run not in TRANSFORMER_RUNS:
            raise ValueError(f"Unknown transformer run {run!r}; choose from {sorted(TRANSFORMER_RUNS)}")
        source = dict(subfolder=TRANSFORMER_RUNS[run], revision=TRANSFORMER_REVISION)
        self.torch = torch
        self.name = f"transformer:{run}"
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_REPO, **source)
        self.model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_REPO, **source)
        self.model.to(self.device).eval()

    def predict_proba(self, texts):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(list(texts[start : start + self.batch_size]), truncation=True,
                                     max_length=512, padding=True, return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(out)


@lru_cache(maxsize=None)
def load_model(name):
    """Build a model from its registry name, e.g. ``bilstm`` or ``graphsage:amt``."""
    kind, _, arg = name.partition(":")
    if kind == "tfidf_lr":
        return TfidfLR((arg,) if arg else DATASETS)
    if kind == "bilstm":
        return BiLSTMModel()
    if kind == "graphsage":
        return GraphSAGEModel(arg or "multifc")
    if kind == "transformer":
        return TransformerModel(run=arg or "roberta_adamw")
    raise ValueError(f"Unknown model: {name}")


# Models trained on all four datasets: compared like-for-like in the main analysis.
POOLED_MODELS = ("tfidf_lr", "bilstm", "graphsage:pooled")
# Single-source models: used for the cross-domain transfer analysis.
SOURCE_MODELS = tuple(f"{kind}:{d}" for kind in ("tfidf_lr", "graphsage") for d in DATASETS)
