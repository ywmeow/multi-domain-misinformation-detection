"""CPU inference adapters; no training data or network access is required.

The RoBERTa weights are not stored in Git. ``python dashboard/download_transformer.py``
fetches them once into the local Hugging Face cache; the app itself never downloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
import json
import logging
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
MAX_CHARACTERS = 20_000
INFERENCE_LOCK = RLock()
LOGGER = logging.getLogger(__name__)

MODELS = ("bilstm", "graphsage", "transformer")
# Trung's fine-tuned roberta-base (Muon optimizer), pinned to one repository commit
# so every teammate loads identical weights.
TRANSFORMER_REPO = "ngqtrung/ee6405-misinfo-detection"
TRANSFORMER_SUBFOLDER = "models/roberta-base-muon"
TRANSFORMER_REVISION = "7837df88331f0b16f9bcbda8fc5e0e4f859a8ccc"
TRANSFORMER_MAX_TOKENS = 512


class ModelUnavailable(RuntimeError):
    """A model that is optional and not set up on this machine (not an error)."""


@dataclass(frozen=True)
class Prediction:
    model: str
    status: str
    label: str | None = None
    p_fake: float | None = None
    threshold: float | None = None
    elapsed_ms: float | None = None
    message: str | None = None
    truncated: bool = False


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a headline, claim, or article to analyze.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"Keep the input under {MAX_CHARACTERS:,} characters.")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _module(name: str, path: Path):
    """Load distinct model.py modules without changing sys.path or sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model definition: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BiLSTMDetector:
    def __init__(self, checkpoint_dir: Path | None = None):
        import spacy
        import torch

        checkpoint_dir = (
            checkpoint_dir or ROOT / "sequential/checkpoints/bilstm_glove100_seed42"
        )
        self.torch = torch
        # The app is CPU-only. Avoid spawning a large thread pool for one headline.
        torch.set_num_threads(4)
        self.vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
        settings = json.loads(
            (checkpoint_dir / "tokenizer_settings.json").read_text(encoding="utf-8")
        )
        checkpoint = torch.load(
            checkpoint_dir / "best.pt", map_location="cpu", weights_only=True
        )
        config = checkpoint["model_config"]
        if (
            len(self.vocab) != config["vocab_size"]
            or set(self.vocab.values()) != set(range(config["vocab_size"]))
            or self.vocab.get("<PAD>") != config["pad_id"]
            or "<UNK>" not in self.vocab
            or settings.get("tokenizer") != 'spacy.blank("en")'
            or settings.get("vocab_size") != config["vocab_size"]
        ):
            raise ValueError("The BiLSTM checkpoint and tokenizer files do not match.")
        self.max_length = settings["max_length"]
        if not isinstance(self.max_length, int) or self.max_length < 1:
            raise ValueError("The saved tokenizer length is invalid.")
        self.tokenizer = spacy.blank("en")
        model_module = _module("dashboard_bilstm_model", ROOT / "sequential/model.py")
        self.model = model_module.BiLSTM(**config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    threshold = 0.5

    def _ids(self, text: str) -> list[int]:
        unknown = self.vocab["<UNK>"]
        tokens = self.tokenizer(text)
        return [
            self.vocab.get(token.text, unknown)
            for token in tokens[:self.max_length]
        ] or [unknown]

    def predict(self, text: str) -> tuple[float, float, bool]:
        tokens = self.tokenizer(text)
        unknown = self.vocab["<UNK>"]
        ids = [
            self.vocab.get(token.text, unknown)
            for token in tokens[:self.max_length]
        ] or [unknown]
        with self.torch.inference_mode():
            inputs = self.torch.tensor([ids], dtype=self.torch.long)
            lengths = self.torch.tensor([len(ids)], dtype=self.torch.long)
            probability = self.model(inputs, lengths).softmax(dim=1)[0, 1].item()
        return probability, self.threshold, len(tokens) > self.max_length

    def predict_batch(self, texts: list[str], batch_size: int = 64):
        """Fake-class scores for many texts (used by LIME's perturbations)."""
        import numpy as np

        torch = self.torch
        pad = self.vocab["<PAD>"]
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [self._ids(text) for text in texts[start:start + batch_size]]
                lengths = torch.tensor([len(ids) for ids in batch], dtype=torch.long)
                inputs = torch.full((len(batch), int(lengths.max())), pad, dtype=torch.long)
                for row, ids in enumerate(batch):
                    inputs[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                scores.append(self.model(inputs, lengths).softmax(dim=1)[:, 1].numpy())
        return np.concatenate(scores)


class GraphSAGEDetector:
    def __init__(self, checkpoint_path: Path | None = None):
        import joblib

        checkpoint_path = checkpoint_path or ROOT / "graph/Models/jruvika_graphsage.joblib"
        self.bundle = joblib.load(checkpoint_path)
        if self.bundle.get("architecture") != "two-layer cosine-weighted mean GraphSAGE":
            raise ValueError("The selected file is not a compatible GraphSAGE checkpoint.")
        if self.bundle.get("dataset") != "jruvika":
            raise ValueError("This dashboard expects the Jruvika GraphSAGE checkpoint.")
        self.graph = _module("dashboard_graphsage_model", ROOT / "graph/model.py")
        # The current Jruvika artifact uses 0.55, selected on development data.
        self.threshold = float(self.bundle.get("threshold", 0.5))

    def predict(self, text: str) -> tuple[float, float, bool]:
        return float(self.predict_batch([text])[0]), self.threshold, False

    def predict_batch(self, texts: list[str], batch_size: int = 256):
        """Fake-class scores for many texts (used by LIME's perturbations)."""
        import numpy as np
        from sklearn.preprocessing import normalize

        bundle = self.bundle
        scores = []
        for start in range(0, len(texts), batch_size):
            tfidf = bundle["vectorizer"].transform(texts[start:start + batch_size])
            features = normalize(bundle["svd"].transform(tfidf)).astype("float32")
            adjacency = self.graph.query_adjacency(
                features, bundle["neighbors"], len(bundle["anchors"]), bundle["k"]
            )
            logits = self.graph.numpy_logits(
                features,
                adjacency,
                bundle["anchors"],
                bundle["anchor_hidden"],
                bundle["state"],
            )
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            scores.append((exponentials / exponentials.sum(axis=1, keepdims=True))[:, 1])
        return np.concatenate(scores)


class TransformerDetector:
    """Fine-tuned roberta-base (Muon), loaded only from the local Hugging Face cache."""

    threshold = 0.5

    def __init__(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ModelUnavailable(
                "Install the transformers package to enable RoBERTa."
            ) from error

        self.torch = torch
        source = dict(
            subfolder=TRANSFORMER_SUBFOLDER,
            revision=TRANSFORMER_REVISION,
            local_files_only=True,  # never download from inside the app
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_REPO, **source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                TRANSFORMER_REPO, **source
            )
        except OSError as error:
            raise ModelUnavailable(
                "RoBERTa weights are not on this machine. "
                "Run python dashboard/download_transformer.py once."
            ) from error
        self.model.eval()

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=TRANSFORMER_MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        )

    def predict(self, text: str) -> tuple[float, float, bool]:
        # Encode one token past the limit: enough to detect truncation without a warning.
        encoded = self.tokenizer(text, truncation=True, max_length=TRANSFORMER_MAX_TOKENS + 1)
        truncated = len(encoded["input_ids"]) > TRANSFORMER_MAX_TOKENS
        return float(self.predict_batch([text])[0]), self.threshold, truncated

    def predict_batch(self, texts: list[str], batch_size: int = 16):
        """Fake-class scores for many texts (label 1 = fake, as in training)."""
        import numpy as np

        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                logits = self.model(**self._encode(list(texts[start:start + batch_size]))).logits
                scores.append(logits.float().softmax(dim=-1)[:, 1].numpy())
        return np.concatenate(scores)


@lru_cache(maxsize=len(MODELS))
def load_detector(model: str):
    """Keep trusted local checkpoints loaded across Streamlit reruns/sessions."""
    if model == "bilstm":
        return BiLSTMDetector()
    if model == "graphsage":
        return GraphSAGEDetector()
    if model == "transformer":
        return TransformerDetector()
    raise ValueError(f"Unknown model: {model}")


def _predict_one(model: str, text: str) -> Prediction:
    """Keep a checkpoint failure from hiding the other models' results."""
    try:
        detector = load_detector(model)
        started = perf_counter()
        probability, threshold, truncated = detector.predict(text)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The checkpoint returned an invalid score.")
        return Prediction(
            model=model,
            status="ready",
            label="fake" if probability >= threshold else "real",
            p_fake=probability,
            threshold=threshold,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            truncated=truncated,
        )
    except ModelUnavailable as error:
        # Optional model not set up here: show how to enable it, not an error.
        return Prediction(model=model, status="unavailable", message=str(error))
    except Exception as error:
        LOGGER.exception("Prediction failed for %s", model)
        if isinstance(error, FileNotFoundError):
            message = "A checkpoint file is missing. Check the local model files."
        else:
            message = "This model could not run. Check the app's terminal for details."
        return Prediction(model=model, status="error", message=message)


def analyze(text: str) -> dict:
    """Normalize the input and run every available classifier on CPU."""
    from threadpoolctl import threadpool_limits

    text = normalize_text(text)
    started = perf_counter()
    # Cached PyTorch/spaCy instances are shared across sessions; run them serially.
    with INFERENCE_LOCK, threadpool_limits(limits=4):
        predictions = [_predict_one(model, text) for model in MODELS]

    completed = [p for p in predictions if p.status == "ready"]
    fake_votes = sum(p.label == "fake" for p in completed)
    agreement = None
    if len(completed) >= 2:
        agreement = "agree" if fake_votes in (0, len(completed)) else "disagree"
    return {
        "text": text,
        "predictions": [asdict(p) for p in predictions],
        "agreement": agreement,
        "completed": len(completed),
        "fake_votes": fake_votes,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
    }
