"""Train the BiLSTM, select on dev mean dataset macro-F1, and evaluate test once."""

import argparse
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import pandas as pd
import psutil
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from data import collate_batch
from embeddings import load_glove
from model import BiLSTM


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_split(cache, split):
    columns = ["id", "label_id", "dataset", "domain", "group_id", "split", "token_length", "input_ids"]
    return pd.read_parquet(cache / f"tokenized_{split}.parquet", columns=columns)


def make_loader(frame, batch_size, shuffle=False):
    rows = frame[["input_ids", "label_id"]].to_dict("records")
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


@torch.inference_mode()
def evaluate(model, frame, batch_size, device):
    model.eval()
    probabilities = []
    for batch in make_loader(frame, batch_size):
        logits = model(batch["input_ids"].to(device), batch["lengths"])
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
    return frame.drop(columns="input_ids").assign(p_fake=probabilities)


def metrics(frame):
    labels = frame["label_id"].to_numpy()
    probabilities = frame["p_fake"].to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    real_f1, fake_f1 = f1_score(labels, predicted, labels=[0, 1], average=None, zero_division=0)
    return {
        "n": len(frame),
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "acc": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "f1_real": float(real_f1),
        "f1_fake": float(fake_f1),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
    }


def group_report(frame):
    return {
        "overall": metrics(frame),
        "dataset": {name: metrics(group) for name, group in frame.groupby("dataset")},
        "domain": {
            f"{dataset}/{domain}": metrics(group)
            for (dataset, domain), group in frame.groupby(["dataset", "domain"])
            if len(group) >= 10
        },
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "cache")
    parser.add_argument("--output", type=Path, default=root / "checkpoints/bilstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embedding-dropout", type=float, default=0.3)
    parser.add_argument("--glove", type=Path, help="GloVe text file; dimension must match --embedding-dim")
    parser.add_argument("--embedding-lr", type=float, default=1e-4, help="Embedding learning rate with GloVe")
    parser.add_argument("--freeze-epochs", type=int, default=1, help="Initial frozen epochs with GloVe")
    parser.add_argument("--patience", type=int, default=3, help="Epochs without dev improvement; 0 disables")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.embedding_dim, args.hidden_size, args.threads) < 1:
        parser.error("epochs, batch size, model dimensions, and threads must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.embedding_dropout < 1:
        parser.error("dropout and embedding dropout must be in [0, 1)")
    if min(args.lr, args.embedding_lr) <= 0 or args.weight_decay < 0:
        parser.error("learning rates must be positive and weight decay nonnegative")
    if min(args.freeze_epochs, args.patience) < 0:
        parser.error("freeze epochs and patience must be nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    vocab = json.loads((args.cache / "vocab.json").read_text(encoding="utf-8"))
    tokenizer_settings = json.loads((args.cache / "settings.json").read_text(encoding="utf-8"))
    train, dev = load_split(args.cache, "train"), load_split(args.cache, "dev")
    train_loader = make_loader(train, args.batch_size, shuffle=True)
    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "embedding_dropout": args.embedding_dropout,
        "pad_id": vocab["<PAD>"],
    }
    model = BiLSTM(**model_config)
    coverage = None
    if args.glove:
        weights, coverage = load_glove(args.glove, vocab, args.embedding_dim)
        with torch.no_grad():
            model.embedding.weight.copy_(weights)
        print(f"GloVe vocabulary coverage: {coverage['vocab_coverage']:.1%}", flush=True)
    model.to(device)
    freeze_epochs = args.freeze_epochs if args.glove else 0
    embedding_lr = args.embedding_lr if args.glove else args.lr
    optimizer = torch.optim.AdamW([
        {"params": [model.embedding.weight], "lr": embedding_lr},
        {"params": [p for p in model.parameters() if p is not model.embedding.weight], "lr": args.lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(parameter.numel() for parameter in model.parameters())

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.cache / "vocab.json", args.output / "vocab.json")
    shutil.copyfile(args.cache / "settings.json", args.output / "tokenizer_settings.json")
    write_json(args.output / "config.json", {
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": model_config,
        "tokenizer": tokenizer_settings,
        "embeddings": coverage,
        "versions": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "psutil": psutil.__version__,
        },
    })
    print(
        f"Device: {device}; parameters: {n_params:,}; train: {len(train):,}; dev: {len(dev):,}",
        flush=True,
    )

    history, best_score, stale_epochs = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.embedding.weight.requires_grad_(epoch > freeze_epochs)
        if epoch == freeze_epochs + 1:
            stale_epochs = 0
        print(
            f"Epoch {epoch}: embeddings {'frozen' if epoch <= freeze_epochs else 'trainable'}",
            flush=True,
        )
        model.train()
        total_loss, examples = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["lengths"])
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            examples += len(labels)
            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs}, batch {step}/{len(train_loader)}, "
                    f"loss {total_loss / examples:.4f}",
                    flush=True,
                )

        dev_predictions = evaluate(model, dev, args.batch_size, device)
        report = group_report(dev_predictions)
        selection = sum(group["macro_f1"] for group in report["dataset"].values()) / len(report["dataset"])
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "dev_loss": report["overall"]["loss"],
            "dev_selection_f1": selection,
            "embeddings_frozen": epoch <= freeze_epochs,
            "embedding_lr": embedding_lr,
            "dev": report["dataset"],
            "elapsed_s": time.perf_counter() - start,
        })
        write_json(args.output / "history.json", history)
        print(f"Epoch {epoch}: dev mean dataset macro-F1 {selection:.4f}", flush=True)
        if selection > best_score:
            stale_epochs = 0
            best_score, best_epoch, best_dev = selection, epoch, report
            torch.save({"model": model.state_dict(), "model_config": model_config}, args.output / "best.pt")
            dev_predictions.to_parquet(args.output / "dev_preds.parquet", index=False)
        else:
            stale_epochs += 1
        if args.patience and epoch > freeze_epochs and stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without dev improvement", flush=True)
            break

    train_time = time.perf_counter() - start
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test = load_split(args.cache, "test")
    test_start = time.perf_counter()
    test_predictions = evaluate(model, test, args.batch_size, device)
    test_time = time.perf_counter() - test_start
    test_predictions.to_parquet(args.output / "test_preds.parquet", index=False)
    memory = psutil.Process().memory_info()
    summary = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "selected_train_loss": history[best_epoch - 1]["train_loss"],
        "best_dev_selection_f1": best_score,
        "n_params": n_params,
        "train_time_s": train_time,
        "test_infer_time_s": test_time,
        "hardware": {
            "device": str(device),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "threads": torch.get_num_threads(),
            "system_ram_gb": psutil.virtual_memory().total / 1e9,
            "process_rss_gb": memory.rss / 1e9,
            "peak_process_rss_gb": memory.peak_wset / 1e9 if hasattr(memory, "peak_wset") else None,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
            ),
        },
        "dev": best_dev,
        "test": group_report(test_predictions),
    }
    write_json(args.output / "metrics.json", summary)
    print("TEST", json.dumps(summary["test"]["overall"]), flush=True)
    print(pd.DataFrame(summary["test"]["dataset"]).T.round(4).to_string(), flush=True)
    print(f"Training: {train_time / 60:.1f} min; outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
