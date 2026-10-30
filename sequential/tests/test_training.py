"""Small CPU checks for pretrained embeddings, freezing, and early stopping."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import collate_batch
from embeddings import load_glove
from model import BiLSTM


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.cache_root = (ROOT / "cache").resolve()
        self.cache_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=self.cache_root, prefix="training-check-")
        self.root = Path(self.directory.name).resolve()
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "known": 2, "Known": 3, "Case": 4, "missing": 5}
        self.glove = self.root / "glove.txt"
        self.glove.write_text(
            "known 0.1 0.2 0.3\ncase 0.4 0.5 0.6\nCase 0.7 0.8 0.9\n", encoding="utf-8"
        )

    def tearDown(self):
        assert self.root.is_relative_to(self.cache_root)
        self.directory.cleanup()

    def test_glove_matching_and_validation(self):
        weights, coverage = load_glove(self.glove, self.vocab, 3)
        torch.testing.assert_close(weights[2], torch.tensor([0.1, 0.2, 0.3]))
        torch.testing.assert_close(weights[3], weights[2])
        torch.testing.assert_close(weights[4], torch.tensor([0.7, 0.8, 0.9]))
        self.assertEqual(weights[0].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(coverage["exact_matches"], 2)
        self.assertEqual(coverage["lowercase_matches"], 1)
        self.assertEqual(coverage["vocab_coverage"], 0.75)
        with self.assertRaises(ValueError):
            load_glove(self.glove, self.vocab, 4)
        self.glove.write_text("known nan 0 0\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_glove(self.glove, self.vocab, 3)
        self.glove.write_text("unrelated 0 0 0\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_glove(self.glove, self.vocab, 3)

    def test_freeze_unfreeze_and_padding(self):
        model = BiLSTM(6, embedding_dim=3, hidden_size=4)
        weights, _ = load_glove(self.glove, self.vocab, 3)
        with torch.no_grad():
            model.embedding.weight.copy_(weights)
        optimizer = torch.optim.AdamW([
            {"params": [model.embedding.weight], "lr": 1e-4},
            {"params": [p for p in model.parameters() if p is not model.embedding.weight], "lr": 1e-3},
        ], weight_decay=0.01)
        batch = collate_batch([
            {"input_ids": [2, 3], "label_id": 0},
            {"input_ids": [4, 5, 2], "label_id": 1},
            {"input_ids": [1], "label_id": 1},
        ])
        for frozen in (True, False):
            model.embedding.weight.requires_grad_(not frozen)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["lengths"])
            loss = nn.functional.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            if frozen:
                torch.testing.assert_close(model.embedding.weight, weights, atol=0, rtol=0)
                self.assertIsNone(model.embedding.weight.grad)
            else:
                self.assertGreater((model.embedding.weight - weights).abs().sum(), 0)
                self.assertEqual(model.embedding.weight.grad[0].count_nonzero(), 0)
        self.assertEqual(model.embedding.weight[0].count_nonzero(), 0)
        model.eval()
        with torch.no_grad():
            logits = model(batch["input_ids"], batch["lengths"])
            padded = nn.functional.pad(batch["input_ids"], (0, 5))
            torch.testing.assert_close(logits, model(padded, batch["lengths"]))
            for index, row in enumerate(batch["input_ids"]):
                length = batch["lengths"][index:index + 1]
                single = model(row[:length.item()].unsqueeze(0), length)
                torch.testing.assert_close(logits[index:index + 1], single, atol=1e-6, rtol=1e-5)

    def test_training_stops_and_reloads_best_checkpoint(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        (cache / "settings.json").write_text('{"max_length": 512}', encoding="utf-8")
        for split in ("train", "dev", "test"):
            frame = pd.DataFrame([{
                "id": f"{split}-{i}", "label_id": i % 2,
                "dataset": "a" if i < 4 else "b", "domain": "news",
                "group_id": f"g{i}", "split": split, "token_length": i % 3 + 1,
                "input_ids": [2 + i % 2] * (i % 3 + 1),
            } for i in range(8)])
            frame.to_parquet(cache / f"tokenized_{split}.parquet", index=False)
        output = self.root / "run"
        command = [
            sys.executable, str(ROOT / "train.py"), "--cache", str(cache), "--output", str(output),
            "--glove", str(self.glove), "--embedding-dim", "3", "--hidden-size", "4",
            "--epochs", "10", "--batch-size", "4", "--threads", "1", "--device", "cpu", "--patience", "3",
            "--lr", "1e-30", "--embedding-lr", "1e-30", "--dropout", "0", "--embedding-dropout", "0",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        history = json.loads((output / "history.json").read_text())
        summary = json.loads((output / "metrics.json").read_text())
        self.assertEqual(len(history), 4)
        self.assertEqual([row["embeddings_frozen"] for row in history], [True, False, False, False])
        self.assertEqual(summary["best_epoch"], 1)
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["final_train_loss"], history[-1]["train_loss"])
        self.assertEqual(history[0]["dev_loss"], summary["dev"]["overall"]["loss"])
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        model = BiLSTM(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        expected, _ = load_glove(self.glove, self.vocab, 3)
        torch.testing.assert_close(model.embedding.weight, expected, atol=0, rtol=0)
        for split in ("dev", "test"):
            frame = pd.read_parquet(cache / f"tokenized_{split}.parquet")
            predictions = pd.read_parquet(output / f"{split}_preds.parquet")
            pd.testing.assert_frame_equal(predictions.drop(columns="p_fake"), frame.drop(columns="input_ids"))
        for freeze, patience, epochs, expected_epochs in ((3, 1, 8, 4), (0, 0, 2, 2)):
            with self.subTest(freeze=freeze, patience=patience):
                other_output = self.root / f"run_{freeze}_{patience}"
                subprocess.run(command + [
                    "--output", str(other_output), "--freeze-epochs", str(freeze),
                    "--patience", str(patience), "--epochs", str(epochs),
                ], check=True, capture_output=True, text=True)
                other_history = json.loads((other_output / "history.json").read_text())
                self.assertEqual(len(other_history), expected_epochs)
                self.assertEqual(
                    [row["embeddings_frozen"] for row in other_history],
                    [epoch <= freeze for epoch in range(1, expected_epochs + 1)],
                )


if __name__ == "__main__":
    unittest.main()
