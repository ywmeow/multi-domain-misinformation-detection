"""Tokenize pooled splits and build a training-only vocabulary for the BiLSTM."""

import argparse
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence


PAD_ID = 0
UNK_ID = 1


def build_vocab(texts, nlp, min_freq=2, max_vocab=50_000):
    counts = Counter(token.text for doc in nlp.pipe(texts) for token in doc)
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for token, count in counts.most_common():
        if count < min_freq or len(vocab) == max_vocab + 2:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def tokenize(frame, nlp, vocab, max_length=512):
    lengths, sequences = [], []
    for doc in nlp.pipe(frame["text"]):
        lengths.append(len(doc))
        sequences.append([vocab.get(token.text, UNK_ID) for token in doc[:max_length]] or [UNK_ID])
    frame = frame.copy()
    frame["token_length"] = lengths
    frame["input_ids"] = sequences
    return frame


def diagnostics(frame, max_length):
    report = {}
    for name, group in frame.groupby("dataset", sort=False):
        lengths = group["token_length"]
        total = sum(len(ids) for ids in group["input_ids"])
        unknown = sum(token == UNK_ID for ids in group["input_ids"] for token in ids)
        report[name] = {
            "rows": len(group),
            "mean_length": float(lengths.mean()),
            "p95_length": float(lengths.quantile(0.95)),
            "max_length": int(lengths.max()),
            "truncation_rate": float((lengths > max_length).mean()),
            "unk_rate": unknown / total if total else 0.0,
        }
    return report


def collate_batch(rows):
    sequences = [torch.tensor(row["input_ids"], dtype=torch.long) for row in rows]
    return {
        "input_ids": pad_sequence(sequences, batch_first=True, padding_value=PAD_ID),
        "lengths": torch.tensor([len(ids) for ids in sequences], dtype=torch.long),
        "labels": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--cache", type=Path, default=root / "cache")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=50_000)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if min(args.min_freq, args.max_vocab, args.max_length) < 1:
        parser.error("--min-freq, --max-vocab, and --max-length must be positive")

    nlp = spacy.blank("en")
    train = pd.read_parquet(args.data / "pooled_train.parquet")
    print("Building vocabulary from training text...", flush=True)
    vocab = build_vocab(train["text"], nlp, args.min_freq, args.max_vocab)
    settings = {
        "tokenizer": 'spacy.blank("en")',
        "min_freq": args.min_freq,
        "max_vocab": args.max_vocab,
        "vocab_size": len(vocab),
        "special_tokens": {"<PAD>": PAD_ID, "<UNK>": UNK_ID},
        "max_length": args.max_length,
        "truncation": "head",
        "padding": "longest_in_batch",
        "empty_text": "<UNK>",
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("pandas", "pyarrow", "spacy", "torch")},
        },
    }
    args.cache.mkdir(parents=True, exist_ok=True)
    print("Vocabulary:", len(vocab), "tokens (including PAD and UNK)", flush=True)

    reports = {}
    for split in ("train", "dev", "test"):
        frame = train if split == "train" else pd.read_parquet(args.data / f"pooled_{split}.parquet")
        print("Tokenizing", split, flush=True)
        frame = tokenize(frame, nlp, vocab, args.max_length)
        frame.to_parquet(args.cache / f"tokenized_{split}.parquet", index=False)
        print(split, len(frame))
        if split != "test":
            reports[split] = diagnostics(frame, args.max_length)
            print(pd.DataFrame(reports[split]).T.round(4).to_string())

    for name, value in (("vocab", vocab), ("settings", settings), ("diagnostics", reports)):
        (args.cache / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
