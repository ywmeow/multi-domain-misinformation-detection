"""Aggregate results/*/metrics.json into results/SUMMARY.md + cross-domain matrix figure."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path("/scratch/users/astar/ares/suns1/trung/nlp/results")
DS = ["amt", "multifc", "mc_fake", "jruvika"]
runs = {d.name: json.load(open(d / "metrics.json")) for d in sorted(R.iterdir()) if (d / "metrics.json").exists()}
L = ["## Pooled multi-domain training (all 4 datasets), test macro-F1\n",
     "| run | model | optimizer | lr | best ep | overall | " + " | ".join(DS) + " | train min | peak GB | tok/s |", "|" + "---|" * 13]
for n, r in runs.items():
    if n.startswith("xd_") or n == "tfidf_lr": continue
    a = r["args"]; t = r["test"]
    L.append(f"| {n} | {a['model']} | {a['optim']} | {a['lr']} | {r['best_epoch']} | {t['overall']['macro_f1']:.4f} | " +
             " | ".join(f"{t['dataset'][d]['macro_f1']:.4f}" for d in DS) + f" | {r['train_time_s']/60:.1f} | {r['peak_gpu_mem_GB']} | {r['train_tokens_per_s']} |")
if "tfidf_lr" in runs or (R / "tfidf_lr/metrics.json").exists():
    b = json.load(open(R / "tfidf_lr/metrics.json"))
    L.append(f"| tfidf_lr (reference) | TF-IDF+LR | - | - | - | {b['overall']:.4f} | " + " | ".join(f"{b[d]:.4f}" for d in DS) + " | - | - | - |")
M = np.full((4, 4), np.nan)
for i, s in enumerate(DS):
    if f"xd_{s}" in runs:
        for j, t in enumerate(DS): M[i, j] = runs[f"xd_{s}"]["test"]["dataset"][t]["macro_f1"]
L += ["\n## Cross-domain transfer (roberta-base, train on row, test on column, test macro-F1)\n", "| train \\ test | " + " | ".join(DS) + " |", "|---|" + "---|" * 4]
for i, s in enumerate(DS): L.append(f"| {s} | " + " | ".join(f"{M[i,j]:.3f}" for j in range(4)) + " |")
fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(M, vmin=.3, vmax=1, cmap="viridis")
ax.set_xticks(range(4), DS, rotation=30); ax.set_yticks(range(4), DS); ax.set_xlabel("test"); ax.set_ylabel("train")
for i in range(4):
    for j in range(4): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] < .7 else "k")
plt.colorbar(im); plt.title("roberta-base single-source transfer (macro-F1)"); plt.tight_layout(); plt.savefig(R / "cross_domain_matrix.png", dpi=130)
(R / "SUMMARY.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
