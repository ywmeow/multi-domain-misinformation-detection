"""Fine-tune a transformer encoder for binary misinformation detection (0=real, 1=fake).

Single GPU, bf16 autocast, custom loop so the optimiser (AdamW or Muon+AdamW hybrid)
is swappable. Model selection uses dev macro-F1 averaged over the *training* datasets
only; every test set is always evaluated, giving a cross-domain row per run.
"""
import argparse, json, os, pathlib, random, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup



def zeropower_ns(G, steps=5):
    """Quintic Newton-Schulz orthogonalisation (Keller Jordan's Muon), plain matmuls in bf16."""
    a_, b_, c_ = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a_ * X + (b_ * A + c_ * (A @ A)) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalised update for 2-D weights. The update is rescaled by 0.2*sqrt(max(m,n))
    so its RMS matches AdamW and the same lr scale can be used (Moonlight convention).
    torch.optim.Muon (2.9) hit 'CUDA unspecified launch failure' in its addmm Gram step on this node."""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p.grad))
                buf.lerp_(p.grad, 1 - g["momentum"])
                upd = zeropower_ns(p.grad.lerp(buf, g["momentum"]), g["ns_steps"]) * (0.2 * max(p.shape) ** 0.5)
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"])


ROOT = pathlib.Path("/scratch/users/astar/ares/suns1/trung/nlp")
DATASETS = ["amt", "multifc", "mc_fake", "jruvika"]

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--model", default="roberta-base")
p.add_argument("--optim", choices=["adamw", "muon"], default="adamw")
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--head_lr", type=float, default=None, help="AdamW lr for non-Muon params when --optim muon")
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--warmup", type=float, default=0.06)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--train_sets", default=",".join(DATASETS))
p.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
p.add_argument("--no_save", action="store_true")
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
train_sets = a.train_sets.split(",")
out_dir = ROOT / "results" / a.name
ckpt_dir = ROOT / "ckpt" / a.name
out_dir.mkdir(parents=True, exist_ok=True)

data = {s: pd.read_parquet(ROOT / "data" / f"pooled_{s}.parquet") for s in ["train", "dev", "test"]}
data["train"] = data["train"][data["train"].dataset.isin(train_sets)].reset_index(drop=True)
data["dev"] = data["dev"][data["dev"].dataset.isin(train_sets)].reset_index(drop=True)

tok = AutoTokenizer.from_pretrained(a.model)


def encode(df, tag):
    cache = ROOT / "data" / f"tok_{a.model.replace('/', '_')}_{a.max_len}_{tag}.pt"
    key = len(df)
    if cache.exists():
        c = torch.load(cache)
        if c["key"] == key and c["ids"] == df.id.tolist():
            return c["input_ids"]
    ids = tok(df.text.tolist(), truncation=True, max_length=a.max_len)["input_ids"]
    torch.save({"key": key, "ids": df.id.tolist(), "input_ids": ids}, cache)
    return ids


enc = {k: encode(v, k + "_" + ("all" if k == "test" else "-".join(train_sets))) for k, v in data.items()}
# test is encoded for all datasets (not filtered)
data["test"] = data["test"].reset_index(drop=True)
print({k: len(v) for k, v in data.items()}, flush=True)
pad_id = tok.pad_token_id


def collate(idx, ids, labels):
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for j, i in enumerate(idx):
        x[j, : len(ids[i])] = torch.tensor(ids[i]); m[j, : len(ids[i])] = 1
    return x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), torch.tensor(labels[idx], dtype=torch.long).to(dev)


model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
n_params = sum(p.numel() for p in model.parameters())

# ---- optimiser ----
if a.optim == "adamw":
    nd = ("bias", "LayerNorm", "layernorm", "layer_norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in nd)], "weight_decay": a.wd},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in nd)], "weight_decay": 0.0},
    ]
    opts = [torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-6)]
else:
    # Muon only for 2-D hidden weight matrices; embeddings, norms, biases, head -> AdamW
    muon_p, adam_p = [], []
    for n, p_ in model.named_parameters():
        (muon_p if p_.ndim == 2 and "embeddings" not in n and "classifier" not in n else adam_p).append(p_)
    opts = [
        Muon(muon_p, lr=a.lr, weight_decay=a.wd),
        torch.optim.AdamW(adam_p, lr=a.head_lr or a.lr, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-6),
    ]
    print(f"muon params {sum(x.numel() for x in muon_p)/1e6:.1f}M, adamw params {sum(x.numel() for x in adam_p)/1e6:.1f}M", flush=True)

steps_per_epoch = (len(data["train"]) + a.bs - 1) // a.bs
total = steps_per_epoch * a.epochs
scheds = [get_linear_schedule_with_warmup(o, int(a.warmup * total), total) for o in opts]


def evaluate(df, ids, bs=128):
    model.eval()
    order = np.argsort([len(x) for x in ids])
    probs = np.zeros(len(df))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            x, m, _ = collate(idx, ids, df.label_id.values)
            probs[idx] = F.softmax(model(input_ids=x, attention_mask=m).logits.float(), -1)[:, 1].cpu().numpy()
    model.train()
    return probs


def metrics(y, pr):
    pred = (pr >= 0.5).astype(int)
    r = {"n": int(len(y)), "acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
         "f1_fake": f1_score(y, pred, pos_label=1, zero_division=0), "f1_real": f1_score(y, pred, pos_label=0, zero_division=0),
         "pos_rate": float(y.mean())}
    r["auroc"] = roc_auc_score(y, pr) if 0 < y.sum() < len(y) else None
    return {k: (round(float(v), 4) if v is not None else None) for k, v in r.items()}


def group_report(df, pr):
    rep = {"overall": metrics(df.label_id.values, pr), "dataset": {}, "domain": {}}
    for k, g in df.assign(_p=pr).groupby("dataset"):
        rep["dataset"][k] = metrics(g.label_id.values, g._p.values)
    for k, g in df.assign(_p=pr).groupby(["dataset", "domain"]):
        if len(g) >= 10:
            rep["domain"][f"{k[0]}/{k[1]}"] = metrics(g.label_id.values, g._p.values)
    return rep


# ---- train ----
labels_tr = data["train"].label_id.values
best, best_state, history = -1, None, []
torch.cuda.reset_peak_memory_stats()
t0 = time.time(); step = 0; tok_seen = 0
model.train()
for ep in range(a.epochs):
    perm = np.random.permutation(len(labels_tr))
    run_loss, te = 0.0, time.time()
    for s in range(0, len(perm), a.bs):
        x, m, y = collate(perm[s : s + a.bs], enc["train"], labels_tr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, sc in zip(opts, scheds):
            o.step(); sc.step(); o.zero_grad(set_to_none=True)
        step += 1; tok_seen += int(m.sum()); run_loss += loss.item()
        if step % 100 == 0:
            print(f"ep{ep} step {step}/{total} loss {run_loss/100:.4f} {tok_seen/(time.time()-t0):.0f} tok/s", flush=True)
            run_loss = 0.0
        if a.max_steps and step >= a.max_steps:
            break
    pr = evaluate(data["dev"], enc["dev"])
    rep = group_report(data["dev"], pr)
    sel = float(np.mean([v["macro_f1"] for v in rep["dataset"].values()]))
    history.append({"epoch": ep + 1, "dev_selection_f1": round(sel, 4), "dev": rep["dataset"], "elapsed_s": round(time.time() - t0, 1)})
    print(f"== epoch {ep+1} dev mean macro-F1 {sel:.4f} | " + " ".join(f"{k}:{v['macro_f1']:.3f}" for k, v in rep["dataset"].items()), flush=True)
    if sel > best:
        best = sel; best_epoch = ep + 1
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    if a.max_steps and step >= a.max_steps:
        break
train_time = time.time() - t0

# ---- final test with best-dev weights ----
model.load_state_dict(best_state)
t1 = time.time()
test_pr = evaluate(data["test"], enc["test"])
test_rep = group_report(data["test"], test_pr)
infer_time = time.time() - t1
dev_pr = evaluate(data["dev"], enc["dev"])
data["test"].assign(p_fake=test_pr).drop(columns=["text"]).to_parquet(out_dir / "test_preds.parquet")
data["dev"].assign(p_fake=dev_pr).drop(columns=["text"]).to_parquet(out_dir / "dev_preds.parquet")
summary = {
    "name": a.name, "args": vars(a), "gpu": torch.cuda.get_device_name(0), "n_params_M": round(n_params / 1e6, 1),
    "best_epoch": best_epoch, "best_dev_selection_f1": round(best, 4),
    "train_time_s": round(train_time, 1), "test_infer_time_s": round(infer_time, 1),
    "train_tokens_per_s": round(tok_seen / train_time), "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    "history": history, "test": test_rep,
}
json.dump(summary, open(out_dir / "metrics.json", "w"), indent=1)
if not a.no_save:
    model.save_pretrained(ckpt_dir / "best", safe_serialization=True); tok.save_pretrained(ckpt_dir / "best")
print("TEST", json.dumps({k: v["macro_f1"] for k, v in test_rep["dataset"].items()}), "overall", test_rep["overall"]["macro_f1"],
      f"| train {train_time/60:.1f} min", flush=True)
