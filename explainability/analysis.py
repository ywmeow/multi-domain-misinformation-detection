"""Run the LIME/SHAP experiments and turn them into tables, plots and an HTML report.

Stages (run from the repository root):

    python explainability/analysis.py explain   # LIME + SHAP on balanced test samples
    python explainability/analysis.py transfer  # single-source models on every target
    python explainability/analysis.py stress    # style-only edits: do predictions flip?
    python explainability/analysis.py report    # tables, plots, examples.html

``explain`` and ``transfer`` append to JSONL files and skip work already done, so an
interrupted run can be resumed by running the same command again.
"""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATASETS, DISPLAY, load_split, sample_split  # noqa: E402
from explainers import deletion_check, explain, highlight_html, top_words  # noqa: E402
from models import load_model  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
EXPLAIN_FILE = RESULTS / "explanations.jsonl"
TRANSFER_FILE = RESULTS / "transfer_explanations.jsonl"

STOPWORDS = None


def stopwords():
    global STOPWORDS
    if STOPWORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOPWORDS = set(ENGLISH_STOP_WORDS) | {"s", "t", "don", "ll", "ve", "re", "m", "d"}
    return STOPWORDS


def models_for(dataset, extra=()):
    """Models that have seen ``dataset`` during training.

    TF-IDF, BiLSTM and the pooled GraphSAGE share the same pooled training data, so they
    form the like-for-like comparison; the single-dataset GraphSAGE is kept alongside.
    """
    return ["tfidf_lr", "bilstm", f"graphsage:{dataset}", "graphsage:pooled", *extra]


def _done_keys(path, fields):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as stream:
        return {tuple(json.loads(line)[f] for f in fields) for line in stream}


def _append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _read(path):
    with open(path, encoding="utf-8") as stream:
        frame = pd.DataFrame([json.loads(line) for line in stream])
    if "threshold" not in frame:
        frame["threshold"] = 0.5
    frame["threshold"] = frame.threshold.fillna(0.5)
    return frame


# --------------------------------------------------------------------------- explain
def run_explain(args):
    fields = ("dataset", "id", "model", "method")
    done = _done_keys(EXPLAIN_FILE, fields)
    extra = tuple(args.extra_models)
    for dataset in args.datasets:
        sample = sample_split(dataset, "test", n=args.n, seed=args.seed)
        for name in models_for(dataset, extra):
            model = load_model(name)
            start, count = time.time(), 0
            for row in sample.itertuples():
                for method in args.methods:
                    if (dataset, row.id, name, method) in done:
                        continue
                    kwargs = {"num_samples": args.lime_samples} if method == "lime" else {"max_evals": args.shap_evals}
                    result = explain(model, row.text, method, max_words=args.max_words, **kwargs)
                    check = deletion_check(model, result["text"], result["weights"], k=5, seed=args.seed)
                    _append(EXPLAIN_FILE, {
                        "dataset": dataset, "id": row.id, "domain": row.domain, "label_id": int(row.label_id),
                        "model": name, "family": model.family, "method": method,
                        "p_fake": result["p_fake"], "threshold": model.threshold, "weights": result["weights"],
                        "local_r2": result.get("local_r2"), "deletion": check, "text": result["text"],
                    })
                    count += 1
            print(f"{DISPLAY[dataset]:8s} {name:20s} {count:3d} new explanations in {time.time() - start:5.0f}s", flush=True)


# --------------------------------------------------------------------------- transfer
def run_transfer(args):
    """Explain single-source models on every target dataset.

    Also scores the single-source TF-IDF models on the full target test sets, which
    gives the traditional baseline the same 4x4 transfer matrix as GraphSAGE/RoBERTa.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import f1_score

    rows = []
    for source in DATASETS:
        model = load_model(f"tfidf_lr:{source}")
        for target in DATASETS:
            test = load_split(target, "test")
            pred = model.predict(list(test.text))
            rows.append({"train": source, "test": target, "macro_f1": f1_score(test.label_id, pred, average="macro")})
    matrix = pd.DataFrame(rows).pivot(index="train", columns="test", values="macro_f1").loc[list(DATASETS), list(DATASETS)]
    matrix.round(4).to_csv(RESULTS / "tfidf_transfer_matrix.csv")
    print("TF-IDF + LR single-source transfer (macro-F1):\n", matrix.round(3), flush=True)

    fields = ("source_model", "target", "id")
    done = _done_keys(TRANSFER_FILE, fields)
    for kind in args.kinds:
        for source in DATASETS:
            name = f"{kind}:{source}"
            model = load_model(name)
            for target in DATASETS:
                sample = sample_split(target, "test", n=args.n, seed=args.seed + 1)
                start, count = time.time(), 0
                for row in sample.itertuples():
                    if (name, target, row.id) in done:
                        continue
                    result = explain(model, row.text, "lime", num_samples=args.lime_samples, max_words=args.max_words)
                    _append(TRANSFER_FILE, {
                        "source_model": name, "kind": kind, "source": source, "target": target, "id": row.id,
                        "label_id": int(row.label_id), "p_fake": result["p_fake"], "threshold": model.threshold,
                        "weights": result["weights"], "text": result["text"],
                    })
                    count += 1
                print(f"{name:20s} -> {DISPLAY[target]:8s} {count:3d} new in {time.time() - start:5.0f}s", flush=True)


# --------------------------------------------------------------------------- report helpers
def aggregate(frame, min_count=3, content_only=True):
    """Global word importance from many local explanations.

    For each word: mean signed weight over the texts that contain it, how many
    texts it appears in, and its share of all attribution mass.
    """
    sums, abs_sums, counts = defaultdict(float), defaultdict(float), defaultdict(int)
    for weights in frame.weights:
        for word, value in weights.items():
            sums[word] += value
            abs_sums[word] += abs(value)
            counts[word] += 1
    total = sum(abs_sums.values()) or 1.0
    out = pd.DataFrame({"word": list(sums), "mean_weight": [sums[w] / counts[w] for w in sums],
                        "n_texts": [counts[w] for w in sums], "mass_share": [abs_sums[w] / total for w in sums]})
    out = out[out.n_texts >= min_count]
    if content_only:
        out = out[~out.word.isin(stopwords()) & ~out.word.str.fullmatch(r"\d+")]
    return out


def top_cues(frame, k=10, **kwargs):
    agg = aggregate(frame, **kwargs)
    score = agg.mean_weight * np.sqrt(agg.n_texts)  # favour words that matter consistently
    agg = agg.assign(score=score)
    fake = agg[agg.mean_weight > 0].nlargest(k, "score")
    real = agg[agg.mean_weight < 0].nsmallest(k, "score")
    return fake, real


def spearman(a, b):
    words = sorted(set(a) | set(b))
    if len(words) < 3:
        return np.nan
    x = pd.Series([a.get(w, 0.0) for w in words]).rank()
    y = pd.Series([b.get(w, 0.0) for w in words]).rank()
    return float(np.corrcoef(x, y)[0, 1])


def topk_overlap(a, b, k=10):
    ta = {w for w, _ in top_words(a, k)}
    tb = {w for w, _ in top_words(b, k)}
    return len(ta & tb) / max(len(ta | tb), 1)


def model_label(name):
    kind, _, arg = name.partition(":")
    if kind == "graphsage":
        # Two GraphSAGE variants are compared, so always say which one.
        return "GraphSAGE (pooled)" if arg == "pooled" else "GraphSAGE (single)"
    return {"tfidf_lr": "TF-IDF+LR", "bilstm": "BiLSTM", "transformer": "Transformer"}.get(kind, kind)


# --------------------------------------------------------------------------- report
def run_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _read(EXPLAIN_FILE)
    frame["family_name"] = frame.model.map(model_label)
    frame["pred"] = (frame.p_fake >= frame.threshold).astype(int)
    frame["correct"] = frame.pred == frame.label_id

    # 1. Global cues per dataset x model (LIME and SHAP).
    cue_rows = []
    for (dataset, model, method), group in frame.groupby(["dataset", "model", "method"]):
        fake, real = top_cues(group, k=args.k)
        for direction, cues in (("fake", fake), ("real", real)):
            for rank, cue in enumerate(cues.itertuples(), 1):
                cue_rows.append({"dataset": dataset, "model": model, "method": method, "direction": direction,
                                 "rank": rank, "word": cue.word, "mean_weight": round(cue.mean_weight, 5),
                                 "n_texts": cue.n_texts})
    cues = pd.DataFrame(cue_rows)
    cues.to_csv(RESULTS / "global_cues.csv", index=False)

    # 2. Agreement: do models point at the same words? Does LIME agree with SHAP?
    keyed = {(r.dataset, r.id, r.model, r.method): r.weights for r in frame.itertuples()}
    agree_rows = []
    for dataset, group in frame.groupby("dataset"):
        for method in group.method.unique():
            names = sorted(group[group.method == method].model.unique())
            for a, b in combinations(names, 2):
                rhos, overlaps = [], []
                for text_id in group.id.unique():
                    wa, wb = keyed.get((dataset, text_id, a, method)), keyed.get((dataset, text_id, b, method))
                    if wa is not None and wb is not None:
                        rhos.append(spearman(wa, wb))
                        overlaps.append(topk_overlap(wa, wb))
                agree_rows.append({"dataset": dataset, "comparison": f"{model_label(a)} vs {model_label(b)}",
                                   "method": method, "spearman": np.nanmean(rhos), "top10_jaccard": np.mean(overlaps),
                                   "n": len(rhos)})
        for model in group.model.unique():
            rhos, overlaps = [], []
            for text_id in group.id.unique():
                wl, ws = keyed.get((dataset, text_id, model, "lime")), keyed.get((dataset, text_id, model, "shap"))
                if wl is not None and ws is not None:
                    rhos.append(spearman(wl, ws))
                    overlaps.append(topk_overlap(wl, ws))
            if rhos:
                agree_rows.append({"dataset": dataset, "comparison": f"{model_label(model)}: LIME vs SHAP",
                                   "method": "both", "spearman": np.nanmean(rhos), "top10_jaccard": np.mean(overlaps),
                                   "n": len(rhos)})
    agreement = pd.DataFrame(agree_rows).round(3)
    agreement.to_csv(RESULTS / "agreement.csv", index=False)

    # 2b. Shortcut audit: is a top cue a genuine signal or a dataset artefact? Count how
    # often it occurs in fake vs real documents of the full test set.
    audit_rows = []
    for dataset in cues.dataset.unique():
        test = load_split(dataset, "test")
        tokens = test.text.str.lower().str.findall(r"\w+").map(set)
        is_fake = test.label_id.to_numpy() == 1
        sub = cues[(cues.dataset == dataset) & (cues.method == "lime") & (cues["rank"] <= 5)]
        for word, group in sub.groupby("word"):
            has = tokens.map(lambda t: word in t).to_numpy()
            p_fake, p_real = has[is_fake].mean(), has[~is_fake].mean()
            audit_rows.append({"dataset": dataset, "word": word, "cue_for": group.direction.mode()[0],
                               "models": ", ".join(sorted(group.model.map(model_label).unique())),
                               "in_fake_docs": round(p_fake, 3), "in_real_docs": round(p_real, 3),
                               "ratio_fake_to_real": round((p_fake + 1e-3) / (p_real + 1e-3), 2)})
    pd.DataFrame(audit_rows).sort_values(["dataset", "cue_for", "ratio_fake_to_real"]).to_csv(
        RESULTS / "shortcut_audit.csv", index=False)

    # 3. Faithfulness: deleting the explained words vs deleting random words.
    faith = frame.dropna(subset=["deletion"]).copy()
    faith["explained_drop"] = faith.deletion.map(lambda d: d["explained_drop"])
    faith["random_drop"] = faith.deletion.map(lambda d: d["random_drop"])
    def flipped(row):
        if "flipped" in row.deletion:
            return row.deletion["flipped"]
        # Older records: the label flips if the drop crosses the decision threshold.
        if row.p_fake >= row.threshold:
            return row.explained_drop > row.p_fake - row.threshold
        return row.explained_drop >= row.threshold - row.p_fake
    faith["flipped"] = faith.apply(flipped, axis=1)
    faith_table = faith.groupby(["family_name", "method"]).agg(
        explained_drop=("explained_drop", "mean"), random_drop=("random_drop", "mean"),
        flip_rate=("flipped", "mean"), n=("id", "size")).round(3).reset_index()
    faith_table.to_csv(RESULTS / "faithfulness.csv", index=False)

    # 4. How concentrated is each model's evidence? Share of attribution on the top 5 words.
    def top5_share(weights):
        values = sorted((abs(v) for v in weights.values()), reverse=True)
        return sum(values[:5]) / (sum(values) or 1.0)
    frame["top5_share"] = frame.weights.map(top5_share)
    concentration = frame[frame.method == "lime"].groupby(["dataset", "family_name"]).top5_share.mean().unstack().round(3)
    concentration.to_csv(RESULTS / "evidence_concentration.csv")

    # 5. Accuracy on the explained sample (context for the explanations).
    sample_acc = frame[frame.method == "lime"].groupby(["dataset", "family_name"]).correct.mean().unstack().round(3)
    sample_acc.to_csv(RESULTS / "sample_accuracy.csv")

    # Plots: top LIME cues per dataset, one panel per model.
    for dataset in DATASETS:
        sub = cues[(cues.dataset == dataset) & (cues.method == "lime")]
        models = list(dict.fromkeys(sub.model))
        if not models:
            continue
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.6), squeeze=False)
        for ax, model in zip(axes[0], models):
            part = sub[sub.model == model]
            part = pd.concat([part[part.direction == "fake"].head(8), part[part.direction == "real"].head(8)])
            part = part.sort_values("mean_weight")
            colours = ["#e6781e" if v > 0 else "#286edc" for v in part.mean_weight]
            ax.barh(part.word, part.mean_weight, color=colours)
            ax.axvline(0, color="#888", lw=0.8)
            ax.set_title(model_label(model), fontsize=11)
            ax.set_xlabel("mean LIME weight on P(fake)")
            ax.tick_params(axis="y", labelsize=9)
        fig.suptitle(f"{DISPLAY[dataset]}: words pushing towards fake (orange) and real (blue)", fontsize=12)
        fig.tight_layout()
        fig.savefig(RESULTS / f"cues_{dataset}.png", dpi=130)
        plt.close(fig)

    if not faith_table.empty:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        labels = [f"{r.family_name}\n{r.method.upper()}" for r in faith_table.itertuples()]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, faith_table.explained_drop, 0.4, label="delete top-5 explained words", color="#e6781e")
        ax.bar(x + 0.2, faith_table.random_drop, 0.4, label="delete 5 random words", color="#9a9a9a")
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel("drop in predicted-class prob.")
        ax.set_title("Faithfulness: explanations find the words the model relies on")
        ax.set_ylim(0, faith_table.explained_drop.max() * 1.3)
        ax.legend(fontsize=9, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / "faithfulness.png", dpi=130)
        plt.close(fig)

    if TRANSFER_FILE.exists():
        transfer_report(plt)
    write_examples_html(frame)
    print(f"Report written to {RESULTS}")


def transfer_report(plt):
    """Why single-source models fail on other datasets.

    For every source model we take the cue words it relies on in its own domain and
    measure (a) how often they occur in each target dataset, (b) how much of its
    attribution on target texts still lands on those cues, and (c) how much the cue
    sets of different sources overlap.
    """
    frame = _read(TRANSFER_FILE)
    rows, cue_sets = [], {}
    for name, group in frame.groupby("source_model"):
        source = group.source.iloc[0]
        own = group[group.target == source]
        fake, real = top_cues(own, k=25, min_count=2)
        cue_words = set(fake.word) | set(real.word)
        cue_sets[name] = (set(fake.word), set(real.word))
        for target, tgroup in group.groupby("target"):
            texts = load_split(target, "test").text.str.lower()
            sample = texts.sample(min(len(texts), 2000), random_state=0)
            tokens = sample.str.findall(r"\w+").map(set)
            coverage = np.mean([len(t & cue_words) > 0 for t in tokens])
            mean_hits = np.mean([len(t & cue_words) for t in tokens])
            on_cues = np.mean([sum(abs(v) for w, v in ws.items() if w in cue_words) / (sum(abs(v) for v in ws.values()) or 1)
                               for ws in tgroup.weights])
            pred = (tgroup.p_fake >= tgroup.threshold).astype(int)
            rows.append({"source_model": name, "kind": group.kind.iloc[0], "source": source, "target": target,
                         "sample_acc": (pred == tgroup.label_id).mean(), "pred_fake_rate": pred.mean(),
                         "true_fake_rate": tgroup.label_id.mean(),
                         "docs_with_any_cue": coverage, "cue_words_per_doc": mean_hits,
                         "attribution_on_source_cues": on_cues})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "transfer_cue_analysis.csv", index=False)

    overlap_rows = []
    for kind in table.kind.unique():
        names = [n for n in cue_sets if n.startswith(kind)]
        for a, b in combinations(sorted(names), 2):
            (fa, ra), (fb, rb) = cue_sets[a], cue_sets[b]
            overlap_rows.append({"kind": kind, "pair": f"{a.split(':')[1]} / {b.split(':')[1]}",
                                 "fake_cue_jaccard": len(fa & fb) / max(len(fa | fb), 1),
                                 "real_cue_jaccard": len(ra & rb) / max(len(ra | rb), 1),
                                 "flipped_cues": sorted((fa & rb) | (ra & fb))})
    pd.DataFrame(overlap_rows).round(3).to_csv(RESULTS / "transfer_cue_overlap.csv", index=False)

    cue_rows = []
    for name, (fake, real) in cue_sets.items():
        cue_rows.append({"source_model": name, "fake_cues": ", ".join(sorted(fake)), "real_cues": ", ".join(sorted(real))})
    pd.DataFrame(cue_rows).to_csv(RESULTS / "transfer_source_cues.csv", index=False)

    for kind in table.kind.unique():
        sub = table[table.kind == kind]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        panels = ((axes[0], "attribution_on_source_cues", "Share of LIME weight on source-domain cues", "Blues", 0, 0.3),
                  (axes[1], "pred_fake_rate", "Predicted-fake rate (balanced sample; 0.5 = unbiased)", "RdBu_r", 0, 1))
        for ax, column, title, cmap, vmin, vmax in panels:
            grid = sub.pivot(index="source", columns="target", values=column).loc[list(DATASETS), list(DATASETS)]
            im = ax.imshow(grid.values, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(4), [DISPLAY[d] for d in DATASETS], rotation=30)
            ax.set_yticks(range(4), [DISPLAY[d] for d in DATASETS])
            ax.set_xlabel("test (target)")
            ax.set_ylabel("train (source)")
            for i in range(4):
                for j in range(4):
                    v = grid.values[i, j]
                    dark = (v - vmin) / (vmax - vmin) > 0.65 or (cmap == "RdBu_r" and v < 0.2)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if dark else "black")
            ax.set_title(title, fontsize=10)
        fig.suptitle(f"{model_label(kind)} trained on one dataset: its cues vanish elsewhere and it defaults to one class",
                     fontsize=12)
        fig.savefig(RESULTS / f"transfer_cues_{kind}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def write_examples_html(frame, per_dataset=3):
    """Side-by-side highlighted explanations for a few texts per dataset."""
    parts = ["<html><head><meta charset='utf-8'><title>LIME / SHAP examples</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;line-height:1.5}"
             ".box{border:1px solid #ddd;border-radius:6px;padding:8px;margin:6px 0}"
             "h3{margin-bottom:2px}.meta{color:#555;font-size:13px}</style></head><body>",
             "<h1>Word-level explanations</h1><p>Orange words push the model towards <b>fake</b>, blue words "
             "towards <b>real</b>; darker means stronger. Hover a word to see its weight. "
             "Texts are cut to the first 256 words.</p>"]
    for dataset in DATASETS:
        sub = frame[frame.dataset == dataset]
        if sub.empty:
            continue
        parts.append(f"<h2>{DISPLAY[dataset]}</h2>")
        ids = sub.drop_duplicates("id").sort_values("label_id").groupby("label_id").head(2).id.tolist()[:per_dataset + 1]
        for text_id in ids:
            rows = sub[sub.id == text_id]
            label = ["real", "fake"][int(rows.label_id.iloc[0])]
            parts.append(f"<h3>{html_escape(text_id)}</h3><div class='meta'>true label: <b>{label}</b></div>")
            for row in rows.sort_values(["model", "method"]).itertuples():
                pred = "fake" if row.p_fake >= row.threshold else "real"
                parts.append(f"<div class='box'><div class='meta'><b>{model_label(row.model)}</b> "
                             f"({row.method.upper()}) — P(fake) = {row.p_fake:.2f} → predicts <b>{pred}</b></div>"
                             f"{highlight_html(row.text, row.weights)}</div>")
    parts.append("</body></html>")
    (RESULTS / "examples.html").write_text("\n".join(parts), encoding="utf-8")


def html_escape(text):
    import html
    return html.escape(str(text))


# --------------------------------------------------------------------------- stress test
EDITS = {
    # Newswire style, built from the strongest "real" cues found by LIME.
    "add_newswire_byline": lambda t: f"WASHINGTON (Reuters) - A spokesman said in a statement on Tuesday: {t}",
    # Blog footer, built from the strongest "fake" cues found on Jruvika (put first so
    # models that truncate long articles still see it).
    "add_blog_source_line": lambda t: f"Source: http://www.conservativedailynews.com news {t}",
}


def run_stress(args):
    """Counterfactual check of the shortcut hypothesis on the full test sets.

    If the models rely on style/publisher cues rather than content, adding a few
    words that do not change any factual claim should move many predictions.
    """
    rows = []
    for dataset in args.datasets:
        test = load_split(dataset, "test")
        for name in models_for(dataset):
            model = load_model(name)
            base = model.predict(list(test.text)).astype(bool)
            for edit, fn in EDITS.items():
                edited = model.predict([fn(t) for t in test.text]).astype(bool)
                for label, label_name in ((1, "fake"), (0, "real")):
                    mask = test.label_id.to_numpy() == label
                    rows.append({"dataset": dataset, "model": model_label(name), "edit": edit,
                                 "true_label": label_name, "n": int(mask.sum()),
                                 "pred_fake_before": round(base[mask].mean(), 3),
                                 "pred_fake_after": round(edited[mask].mean(), 3),
                                 "changed": round((base[mask] != edited[mask]).mean(), 3)})
            print(f"{DISPLAY[dataset]:8s} {name:20s} done", flush=True)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "stress_test.csv", index=False)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("explain", help="LIME/SHAP on in-domain models")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--methods", nargs="+", default=["lime", "shap"], choices=["lime", "shap"])
    p.add_argument("--n", type=int, default=30, help="texts per dataset (half real, half fake)")
    p.add_argument("--lime-samples", type=int, default=500)
    p.add_argument("--shap-evals", type=int, default=500)
    p.add_argument("--max-words", type=int, default=256)
    p.add_argument("--extra-models", nargs="*", default=[], help="e.g. transformer:roberta_adamw")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=run_explain)

    p = sub.add_parser("transfer", help="single-source models explained on every target")
    p.add_argument("--kinds", nargs="+", default=["tfidf_lr", "graphsage"], choices=["tfidf_lr", "graphsage"])
    p.add_argument("--n", type=int, default=20, help="texts per target dataset")
    p.add_argument("--lime-samples", type=int, default=300)
    p.add_argument("--max-words", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=run_transfer)

    p = sub.add_parser("stress", help="add style-only phrases and count flipped predictions")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.set_defaults(func=run_stress)

    p = sub.add_parser("report", help="tables, plots and examples.html")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
