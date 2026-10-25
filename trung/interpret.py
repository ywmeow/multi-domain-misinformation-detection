"""Interpretation of trained checkpoints.

1. Slice analysis: metrics by dataset / domain / length bucket, calibration (ECE), model agreement.
2. Token attribution: Integrated Gradients (captum) over the embedding layer -> top tokens per
   dataset x predicted class, to look for shortcut cues (source artefacts, boilerplate).
3. Representation: PCA of [CLS]-layer embeddings coloured by dataset and label.
Usage: python interpret.py --run roberta_adamw [--others deberta_adamw roberta_muon_lr1e-4]
"""
import argparse, collections, json, pathlib, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path("/scratch/users/astar/ares/suns1/trung/nlp")
p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--others", nargs="*", default=[])
p.add_argument("--n_attr", type=int, default=40, help="examples per dataset for IG")
p.add_argument("--max_len", type=int, default=512)
a = p.parse_args()
out = ROOT / "results" / a.run / "interp"
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")

test = pd.read_parquet(ROOT / "data/pooled_test.parquet")
pr = pd.read_parquet(ROOT / "results" / a.run / "test_preds.parquet")
assert (pr.id.values == test.id.values).all()
test["p_fake"] = pr.p_fake.values
test["pred"] = (test.p_fake >= 0.5).astype(int)
tok = AutoTokenizer.from_pretrained(ROOT / "ckpt" / a.run / "best")
test["n_tok"] = [len(x) for x in tok(test.text.tolist(), truncation=False)["input_ids"]]
report = {}

# ---------- 1. slices ----------
def mf1(g):
    return f1_score(g.label_id, g.pred, average="macro") if g.label_id.nunique() > 1 else np.nan

test["len_bucket"] = pd.cut(test.n_tok, [0, 32, 128, 256, 512, 10**6], labels=["<=32", "33-128", "129-256", "257-512", ">512 (truncated)"])
lb = test.groupby(["dataset", "len_bucket"], observed=True).apply(
    lambda g: pd.Series({"n": len(g), "acc": (g.pred == g.label_id).mean(), "macro_f1": mf1(g)}), include_groups=False).round(3)
lb.to_csv(out / "by_length.csv")


def ece(y, pf, bins=10):
    conf = np.where(pf >= 0.5, pf, 1 - pf); correct = ((pf >= 0.5).astype(int) == y)
    e, edges = 0.0, np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

report["ece"] = {k: round(ece(g.label_id.values, g.p_fake.values), 4) for k, g in test.groupby("dataset")}
report["ece"]["overall"] = round(ece(test.label_id.values, test.p_fake.values), 4)

# confidence of errors, FP/FN split per dataset
cm = test.groupby("dataset").apply(lambda g: pd.Series({
    "FP(real->fake)": int(((g.pred == 1) & (g.label_id == 0)).sum()), "FN(fake->real)": int(((g.pred == 0) & (g.label_id == 1)).sum()),
    "n": len(g)}), include_groups=False)
cm.to_csv(out / "error_types.csv")

# model agreement
if a.others:
    names = [a.run] + a.others
    P = {n: (pd.read_parquet(ROOT / "results" / n / "test_preds.parquet").p_fake.values >= 0.5).astype(int) for n in names}
    y = test.label_id.values
    agree = pd.DataFrame({m: {n: float((P[m] == P[n]).mean()) for n in names} for m in names}).round(4)
    agree.to_csv(out / "model_agreement.csv")
    wrong = np.stack([P[n] != y for n in names])
    report["errors_shared_by_all"] = int(wrong.all(0).sum()); report["errors_any"] = int(wrong.any(0).sum())
    report["oracle_acc_any_model_right"] = round(float(1 - wrong.all(0).mean()), 4)
    maj = (np.stack([P[n] for n in names]).mean(0) >= 0.5).astype(int)
    report["majority_vote"] = {k: round(f1_score(g.label_id, maj[g.index], average="macro"), 4) for k, g in test.groupby("dataset")}

# ---------- model ----------
model = AutoModelForSequenceClassification.from_pretrained(ROOT / "ckpt" / a.run / "best").to(dev).eval()
emb_layer = model.get_input_embeddings()
pad = tok.pad_token_id


def batch_ids(text):
    ids = tok(text, truncation=True, max_length=a.max_len, return_tensors="pt")["input_ids"].to(dev)
    return ids


def fwd(ids, mask=None):
    return torch.softmax(model(input_ids=ids, attention_mask=(ids != pad).long()).logits.float(), -1)


# ---------- 2. Integrated Gradients ----------
lig = LayerIntegratedGradients(lambda ids: fwd(ids)[:, 1], emb_layer)
tok_stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))  # (dataset,pred)->tok->[sum,count]
examples = []
rng = np.random.RandomState(0)
sel = []
for ds, g in test.groupby("dataset"):
    k = min(a.n_attr, len(g))
    sel += list(rng.choice(g.index, size=k, replace=False))
special = set(tok.all_special_tokens)
for i in sel:
    row = test.loc[i]
    ids = batch_ids(row.text)
    base = torch.full_like(ids, pad)
    base[:, 0] = ids[:, 0]; base[:, -1] = ids[:, -1]
    attr = lig.attribute(ids, baselines=base, n_steps=24, internal_batch_size=24)
    score = attr.sum(-1)[0]
    score = (score / (score.abs().sum() + 1e-9)).detach().cpu().numpy()  # signed share toward "fake"
    toks = tok.convert_ids_to_tokens(ids[0].cpu())
    for t, s in zip(toks, score):
        if t in special: continue
        t = t.replace("Ġ", "").replace("▁", "").lower()
        if not re.search(r"[a-z]{2}", t) or t in ENGLISH_STOP_WORDS: continue  # content words only
        d = tok_stats[(row.dataset, int(row.pred))][t]; d[0] += float(s); d[1] += 1
    top = np.argsort(-np.abs(score))[:8]
    examples.append({"id": row.id, "dataset": row.dataset, "domain": row.domain, "label": int(row.label_id), "pred": int(row.pred),
                     "p_fake": round(float(row.p_fake), 3), "top_tokens": [(toks[j], round(float(score[j]), 3)) for j in top],
                     "text_head": row.text[:240]})
json.dump(examples, open(out / "ig_examples.json", "w"), indent=1)

rows = []
for (ds, pred), d in tok_stats.items():
    items = [(t, v[0], v[1]) for t, v in d.items() if v[1] >= 2]
    fake_push = sorted(items, key=lambda x: -x[1])[:12]   # most push toward fake
    real_push = sorted(items, key=lambda x: x[1])[:12]    # most push toward real
    rows.append({"dataset": ds, "pred_class": "fake" if pred else "real",
                 "toward_fake": ", ".join(f"{t}({s:+.2f})" for t, s, c in fake_push),
                 "toward_real": ", ".join(f"{t}({s:+.2f})" for t, s, c in real_push)})
pd.DataFrame(rows).sort_values(["dataset", "pred_class"]).to_csv(out / "ig_top_tokens.csv", index=False)

# ---------- 3. representation ----------
sub = test.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(400, len(g)), random_state=0), include_groups=False)
sub["dataset"] = test.loc[sub.index, "dataset"]
embs = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in sub.index:
        ids = batch_ids(test.loc[i, "text"])
        h = model(input_ids=ids, attention_mask=(ids != pad).long(), output_hidden_states=True).hidden_states[-1][0, 0]
        embs.append(h.float().cpu().numpy())
Z = PCA(2, random_state=0).fit_transform(np.stack(embs))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for ds, c in zip(sorted(sub.dataset.unique()), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    m = (sub.dataset == ds).values; ax[0].scatter(Z[m, 0], Z[m, 1], s=6, c=c, label=ds, alpha=.6)
ax[0].legend(); ax[0].set_title("last-layer <s> embedding (PCA) by dataset")
for lab, c in [(0, "tab:green"), (1, "tab:red")]:
    m = (sub.label_id == lab).values; ax[1].scatter(Z[m, 0], Z[m, 1], s=6, c=c, label=["real", "fake"][lab], alpha=.6)
ax[1].legend(); ax[1].set_title("same, by label")
plt.tight_layout(); plt.savefig(out / "pca_embeddings.png", dpi=130); plt.close()

# length plot
fig, ax = plt.subplots(figsize=(7, 4))
for ds, g in lb.reset_index().groupby("dataset"):
    ax.plot(g.len_bucket.astype(str), g.macro_f1, marker="o", label=ds)
ax.set_ylabel("macro-F1"); ax.set_title("test macro-F1 by token length"); ax.legend(); plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(out / "f1_by_length.png", dpi=130); plt.close()

json.dump(report, open(out / "report.json", "w"), indent=1)
print(json.dumps(report, indent=1)); print(lb); print(cm)
