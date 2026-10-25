# trung/ - transformer-base training + interpretation (EE6405 Group 43)

Fine-tunes transformer encoders for binary misinformation detection (0 = real, 1 = fake) on the **pooled** team datasets
(AMT, MultiFC, MC-Fake, Jruvika - the `preprocessing/output/*/binary_{train,dev,test}` splits, unchanged).
Everything ran on **one NVIDIA H100 80GB**, bf16 autocast, max 512 tokens, batch 32, 3 epochs, seed 42.

- Data + 3 checkpoints + results + code, one public repo: https://huggingface.co/ngqtrung/ee6405-misinfo-detection (detailed README there)
- Dataset note: the first discussion doc lists ISOT/CoAID/COVID-19/LIAR; the repo (and these runs) use AMT/MultiFC/MC-Fake/Jruvika.

## Results (test macro-F1; model picked on dev mean macro-F1 over the 4 datasets, test touched once)

| model | optimizer | overall | amt (n=48) | multifc | mc_fake | jruvika | AUROC | solo train time (1xH100) |
|---|---|---|---|---|---|---|---|---|
| TF-IDF 1-2gram + LR (reference) | - | 0.8144 | 0.578 | 0.646 | 0.758 | 0.945 | - | ~1 min (CPU) |
| roberta-base | AdamW lr 2e-5 | 0.8317 | 0.832 | 0.690 | 0.816 | 0.985 | 0.931 | 4.4 min |
| deberta-v3-base | AdamW lr 2e-5 | 0.8337 | 0.851 | 0.705 | 0.811 | 0.975 | 0.934 | 12.2 min |
| roberta-base | **Muon** lr 5e-5 (+AdamW 2e-5 for emb/norm/head) | **0.8391** | 0.895 | 0.708 | 0.817 | 0.988 | 0.933 | 5.7 min |

Full table incl. Muon lr sweep and runs: `results/SUMMARY.md`. Per-run `metrics.json` has per-dataset and per-domain metrics.

**Runtime** (one H100 80GB; "solo" = nothing else on the GPU; token throughput counts non-pad tokens):
roberta-base AdamW 4.4 min (128k tok/s, 8.7 GB peak), roberta-base Muon 5.7 min (97k tok/s, 8.3 GB), deberta-v3-base 12.2 min (24 GB peak).
Test inference over 5,715 texts: 1.5 s (roberta-base), 5.1 s (deberta-v3-base). The checkpoints on HF came from earlier runs with the same settings that shared the GPU with 2-3 other runs
(10.6 / 21.0 / 18.4 min wall-clock), so their timings are inflated; the solo reruns (`solo_*`) give clean times.

**Noise warning.** Same seed, solo vs shared GPU, gives overall macro-F1 0.8317 vs 0.8360 (roberta AdamW), 0.8391 vs 0.8408 (Muon), 0.8337 vs 0.8349 (deberta) (bf16 non-determinism).
Single seed, so the 0.832-0.839 spread between the three models is **not** a significant ranking. AMT has only 48 test items (one error = ~2 points).
The Muon lr sweep (2e-5 / 5e-5 / 1e-4: 0.835 / 0.839 / 0.835) shows Muon is at least as good as AdamW here, not clearly better.

## Muon
`torch.optim.Muon` (torch 2.9) crashed with `CUDA error: unspecified launch failure` inside its Newton-Schulz `addmm` on this node (reproducible), so `train.py` has a
small own Muon (momentum 0.95 + Nesterov, 5-step quintic Newton-Schulz in bf16, update rescaled by 0.2*sqrt(max(m,n)) to match AdamW RMS).
Muon is applied to the 2-D hidden weights of the encoder (84.9M params); embeddings, LayerNorm, biases and the classifier head use AdamW (39.7M).

## Interpretation (files in `results/<run>/interp/`)
1. **Binary accuracy hides class-specific failure.** Fake-class F1 vs real-class F1 (roberta AdamW): mc_fake 0.68 / 0.95 (only 16% fake), multifc 0.80 / 0.58 (68% fake). The weak class is always the minority one;
   errors are mostly missed fakes in mc_fake (176 FN vs 73 FP) and balanced in multifc (323 FN / 336 FP). Per-domain macro-F1 (roberta): mc_fake politics 0.95, health 0.81, entertainment 0.81, conflict 0.61; multifc snes 0.58, pomt 0.66.
2. **Length.** Long articles (jruvika, mc_fake > 512 tokens, truncated to the head) are not harder (jruvika 0.97 acc, mc_fake 0.92). Short multifc claims are the hard case: acc 0.74 for <=32 tokens, 0.64 for 33-128 tokens (claim-only text carries little evidence; this is a claim-plausibility task).
3. **Shortcut evidence (Integrated Gradients over embeddings, 80 test texts per dataset, content words; `ig_top_tokens.csv`).**
   Jruvika: predicting *fake* is pushed by `news`, `readers`, `source`, `staff`, `www`; *real* by `reuters`, `caption`, `image`, `copyright` (BBC image-credit boilerplate), `said`.
   MC-Fake: *real* by `said`, `according`, `cnbc`, `china`, `economic`; *fake* by celebrity/gossip names and topics (`divorce`, `pitt`, `taylor`, `beyon`, `instagram`).
   So the models partly learn publisher style and topic, not veracity. This fits the 0.99 jruvika score (random topic-matched articles), the TF-IDF baseline already reaching 0.945 there, and the transfer matrix below.
4. **Cross-domain transfer is poor** (single-source roberta-base, train row -> test column, macro-F1; `results/cross_domain_matrix.png`):

   | train \ test | amt | multifc | mc_fake | jruvika |
   |---|---|---|---|---|
   | amt | 0.916 | 0.484 | 0.632 | 0.852 |
   | multifc | 0.604 | 0.704 | 0.459 | 0.596 |
   | mc_fake | 0.459 | 0.471 | 0.809 | 0.470 |
   | jruvika | 0.531 | 0.351 | 0.639 | 0.993 |

   Off-diagonal values are near chance (0.35-0.64) except amt -> jruvika (0.85) (both news-article style). Pooling all four fixes most of this (0.69-0.99 on every set), i.e. multi-domain training helps mainly by seeing each domain's cues.
   MultiFC <-> news-article transfer fails because claims and articles differ in length, register and label meaning (MultiFC fake = fact-checker rating).
5. **Representation** (`pca_embeddings.png`): the first PCA axis of the last-layer `<s>` vector is the fake/real axis (right panel); datasets occupy different parts of it (left panel), jruvika/mc_fake form tight clusters at the ends, multifc and amt sit in the uncertain middle.
6. **Calibration / agreement.** ECE 0.02 (jruvika, mc_fake), 0.09 (multifc), 0.15 (amt, tiny n); overall 0.052. The three models agree on 89-92% of test items; 490 items are wrong for all three, 1,346 for at least one; a majority vote of the three gives per-dataset macro-F1 amt 0.873 / multifc 0.712 / mc_fake 0.814 / jruvika 0.988, i.e. about the best single model, no real ensemble gain. Shared errors point to label noise / unanswerable claims rather than model capacity.

## Reproduce
```bash
python prep_data.py          # needs the repo's preprocessing/output + extracted mc_fake/jruvika 7z (see below)
python train.py --name roberta_adamw --model roberta-base --optim adamw --lr 2e-5
python train.py --name deberta_adamw --model microsoft/deberta-v3-base --optim adamw --lr 2e-5
python train.py --name roberta_muon_lr5e-5 --optim muon --lr 5e-5 --head_lr 2e-5
python train.py --name xd_mc_fake --train_sets mc_fake --epochs 3 --no_save   # cross-domain; amt: --epochs 10 --bs 16 --lr 3e-5; jruvika: --epochs 5
python baseline_tfidf.py; python interpret.py --run roberta_adamw --others deberta_adamw roberta_muon_lr5e-5 --n_attr 80; python summarize.py
```
`mc_fake` / `jruvika` archives: concatenate `mc_fake.7z.00*` and extract with `py7zr`. Paths are hard-coded to the scratch dir at the top of each script. Deps: torch 2.9, transformers 4.57, captum, py7zr, scikit-learn, pandas.
Data and checkpoints are not in git (HF only). `logs/` holds the training logs, `results/` the metrics, test predictions and interpretation outputs.
