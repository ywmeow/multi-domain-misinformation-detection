"""Predict real/fake labels with a saved BiLSTM and its original vocabulary."""

import argparse
import json
from pathlib import Path
import unicodedata

import spacy
import torch

from model import BiLSTM


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/bilstm_glove100_seed42"


class MisinformationDetector:
    def __init__(self, checkpoint_dir=DEFAULT_CHECKPOINT, device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
        settings = json.loads((checkpoint_dir / "tokenizer_settings.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
        model_config = checkpoint["model_config"]
        size = model_config["vocab_size"]
        if len(self.vocab) != size or set(self.vocab.values()) != set(range(size)):
            raise ValueError("Vocabulary IDs do not match the checkpoint dimensions")
        if self.vocab.get("<PAD>") != model_config["pad_id"] or "<UNK>" not in self.vocab:
            raise ValueError("Vocabulary special tokens do not match the checkpoint")
        if settings["tokenizer"] != 'spacy.blank("en")' or settings["vocab_size"] != size:
            raise ValueError("Tokenizer settings do not match this model")
        self.max_length = settings["max_length"]
        if not isinstance(self.max_length, int) or self.max_length < 1:
            raise ValueError("Tokenizer max_length must be a positive integer")
        self.tokenizer = spacy.blank("en")
        self.model = BiLSTM(**model_config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide a nonempty headline, claim, or article")
        text = " ".join(unicodedata.normalize("NFC", text).split())
        tokens = self.tokenizer(text)
        unknown = self.vocab["<UNK>"]
        ids = [self.vocab.get(token.text, unknown) for token in tokens[:self.max_length]] or [unknown]
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        probability = self.model(inputs, lengths).softmax(dim=1)[0, 1].item()
        label = int(probability >= 0.5)
        return {"label_id": label, "label": "fake" if label else "real", "p_fake": probability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Saved run directory")
    parser.add_argument("--text", required=True, help="Headline, claim, or article to classify")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(4)
    try:
        result = MisinformationDetector(args.checkpoint, args.device).predict(args.text)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
