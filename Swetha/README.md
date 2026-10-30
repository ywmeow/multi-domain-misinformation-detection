# GraphSAGE misinformation detection

Four GraphSAGE models are trained separately on AMT, MultiFC, MC-Fake and Jruvika. Each model is tested on all four datasets, giving 16 train/test pairs. Labels are **0=real** and **1=fake**.

## Results

Accuracy, precision, recall and F1-score are recorded for every train/test pair. Precision, recall and F1 use **macro averaging**: calculate each metric separately for real and fake, then average the two class scores equally. Per-class scores are also retained in the CSV.

Within-dataset results:

| Dataset | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| AMT | 0.6458 | 0.6461 | 0.6458 | 0.6457 |
| MultiFC | 0.6535 | 0.6403 | 0.6588 | 0.6367 |
| MC-Fake | 0.8280 | 0.6967 | 0.7537 | 0.7172 |
| Jruvika | 0.9793 | 0.9795 | 0.9791 | 0.9792 |

Test macro-F1, with training datasets as rows and test datasets as columns:

| Train / test | AMT | MultiFC | MC-Fake | Jruvika |
|---|---:|---:|---:|---:|
| AMT | 0.6457 | 0.5064 | 0.5096 | 0.5997 |
| MultiFC | 0.3778 | 0.6367 | 0.5892 | 0.4856 |
| MC-Fake | 0.4198 | 0.4719 | 0.7172 | 0.5964 |
| Jruvika | 0.4074 | 0.3953 | 0.5244 | 0.9792 |

Jruvika gives the best within-dataset result: **0.9792 macro-F1 and 97.93% accuracy**. Its performance drops to 0.3953–0.5244 macro-F1 on other datasets. The best cross-dataset result is **AMT → Jruvika, 0.5997**. Average macro-F1 is 0.7447 on the diagonal and 0.4903 off the diagonal, showing limited transfer between datasets.

AMT has only 48 test samples, so its scores are sensitive to individual errors. MC-Fake has a large real-class majority; macro-F1 and fake recall are more informative than accuracy alone. Jruvika contains repeated or mismatched article bodies, and MultiFC uses short claims with source-specific ratings. Dataset style and source cues may influence predictions. These are single-seed results, without a statistical-significance claim.

### Which trained model transfers best?

The table below averages each source model's results on the **other three datasets**, excluding its own test set. Each target dataset receives equal weight, regardless of test-set size.

| Training dataset | Mean accuracy | Mean macro precision | Mean macro recall | Mean macro-F1 |
|---|---:|---:|---:|---:|
| AMT | 0.5707 | 0.5608 | 0.5738 | **0.5386** |
| MC-Fake | 0.5511 | 0.7039 | 0.5894 | 0.4960 |
| MultiFC | 0.5696 | 0.5154 | 0.5205 | 0.4842 |
| Jruvika | 0.5836 | 0.5653 | 0.5238 | 0.4424 |

**The AMT-trained model has the best average cross-dataset macro-F1 in this run.** Macro-F1 is the main comparison metric because it gives real and fake equal importance and is less dominated by the majority class than accuracy. Jruvika has the highest mean transfer accuracy, while MC-Fake has the highest mean macro precision; neither leads on macro-F1. The models therefore do not have the same ranking on every metric.

**None of the four models shows strong, consistent cross-dataset performance.** The highest individual transfer macro-F1 is only 0.5997, and source-model transfer averages range from 0.4424 to 0.5386. Jruvika's strong within-dataset result does not carry over: its mean transfer macro-F1 is the lowest of the four. AMT is the best transfer result observed here, not evidence of a reliable general-purpose misinformation detector. Each source is evaluated on a different set of three targets, so this average is descriptive rather than a controlled ranking of source-dataset quality.

The following factors could explain the transfer gap:

- **Different text formats and labels.** MultiFC contains short fact-check claims, whereas the news datasets contain headlines and article bodies. Their label collection procedures and available context also differ.
- **Source-specific features.** Each model learns its own TF-IDF vocabulary and SVD representation. Unseen terms and different word usage in another dataset can weaken its node features. Publisher style, topics or repeated templates may be predictive within a dataset without transferring.
- **Similarity is not factual evidence.** Cosine edges connect documents with similar vocabulary. Articles about the same topic can have different labels; aggregating such neighbors can reinforce an incorrect prediction when the target distribution changes.
- **Class and topic shifts.** The real/fake proportions and topic mixtures differ across datasets. Training-class weighting does not guarantee balanced predictions on a new distribution.
- **No target adaptation.** Checkpoint selection uses source development data only. The models are deliberately tested unchanged on unseen datasets, without fitting a target vocabulary, graph or decision threshold.

These are plausible explanations supported by the experiment setup and known data limitations, not isolated causal findings. The 4×4 evaluation does not determine how much each factor contributes.

All 16 evaluations are in [results/results.csv](results/results.csv). The main columns are `accuracy`, `macro_precision`, `macro_recall` and `macro_f1`. Each row also contains sample counts, balanced accuracy, per-class precision/recall/F1, ROC-AUC, average precision, confusion counts, checkpoint selection, training time and model/data hashes. Confusion columns mean:

- `true_real`: real predicted as real.
- `false_fake`: real predicted as fake.
- `false_real`: fake predicted as real.
- `true_fake`: fake predicted as fake.

## Folder structure

```text
Swetha/
  README.md
  requirements.txt
  requirements-lock.txt
  prep_data.py
  model.py
  train.py
  test.py
  implementation.py
  evaluation_utils.py
  Models/
    amt_graphsage.joblib
    multifc_graphsage.joblib
    mc_fake_graphsage.joblib
    jruvika_graphsage.joblib
  logs/
    prep_data.log
    train.log
    test.log
  results/
    results.csv
    training.json
    split_manifest.json.gz
```

`results.csv` is the only CSV. `training.json` holds the four training configurations and epoch histories. `split_manifest.json.gz` contains IDs, labels, groups, split membership and text hashes, without article text; it is needed to reproduce the evaluated splits.

## Model architecture

### Why GraphSAGE was chosen

The project needs a graph model that can use the same input type across all four datasets. The local inputs provide document text but not a common social-interaction or propagation network. A document-similarity graph supplies a consistent graph representation for all four sources.

GraphSAGE fits this setup because it learns how to combine a document's own features with neighboring document features. Its inductive formulation also allows a new article to query existing training anchors, which suits cross-dataset testing and the interactive implementation. Held-out documents do not need to be added to the training graph or exchange messages with one another.

The two-layer weighted-mean version keeps the implementation manageable on the available laptop GPU. The 128-dimensional inputs and 64-unit hidden layer allow full-neighborhood training without a large language encoder. Two layers use both direct and second-hop neighborhood information; separate self projections retain each document's own features.

```text
Text → TF-IDF (unigrams/bigrams, up to 40,000 features)
     → SVD + normalization (128 node features)
     → cosine-weighted kNN graph (10 training neighbors)
     → GraphSAGE (64 units, ReLU, dropout 0.3)
     → GraphSAGE (2 logits) → softmax
```

Documents are nodes. Both GNN layers learn separate self and neighbor projections by backpropagation. TF-IDF and SVD generate features; they are not separate classifiers. Each source model fits its vocabulary, SVD, graph and neural weights using its own training partition. Source-development macro-F1 selects the checkpoint. Test documents query only source training anchors, without test-to-test messages or target adaptation.

Training uses AdamW, learning rate 0.003, weight decay 0.0001, class-balanced cross entropy and seed 42. It runs for at most 300 epochs, checking dev macro-F1 at epoch 1 and every 5 epochs, and stops after 8 checks without improvement. The recorded run used an NVIDIA GeForce RTX 3050 Laptop GPU. Sparse GPU operations can cause small rerun differences.

## Saved models

| Joblib file | Training data / checkpoint selection |
|---|---|
| `Models/amt_graphsage.joblib` | AMT train / AMT dev |
| `Models/multifc_graphsage.joblib` | MultiFC train / MultiFC dev |
| `Models/mc_fake_graphsage.joblib` | MC-Fake train / MC-Fake dev |
| `Models/jruvika_graphsage.joblib` | Jruvika train / Jruvika dev |

Joblib is the serialization format. Each file contains the fitted vectorizer/SVD, neural weights, neighbor index and numerical training-anchor representations. Four saved models cover all 16 evaluations. Inference uses NumPy calculations checked against the trained PyTorch network; it does not require PyTorch or source datasets. Load only trusted joblib files with compatible dependencies.

## Run inference

Open `implementation.py` and press Run in the IDE. Paste an article or claim and finish with a blank line. The same input runs through all four models, and the script prints each label and fake probability. No arguments are required.

From the repository root, the equivalent command is:

```powershell
python Swetha/implementation.py
```

Probabilities are uncalibrated classifier scores. The predictions do not independently verify factual claims.

## Training and testing

Python 3.10+ and Windows `tar` are required. From the repository root:

```powershell
python -m pip install -r Swetha/requirements.txt
python Swetha/prep_data.py
python Swetha/train.py
python Swetha/test.py
```

`prep_data.py` validates shared source files and split separation. It reads AMT/MultiFC from existing preprocessing CSVs, Jruvika from its existing source CSV, and MC-Fake directly from its original 7z member in memory. No datasets are copied or extracted into this folder.

`train.py` saves four source-specific checkpoints and updates `results/training.json`. It automatically uses CUDA when available; `python Swetha/train.py --device cpu` forces CPU. `test.py` evaluates all 16 pairs and replaces `results/results.csv`. It does not select or tune models using test results. The README records the completed run; subsequent training runs may produce slightly different scores.

Paths resolve from the scripts, so they also run from an IDE. The shared sources and `results/split_manifest.json.gz` must remain available for training/testing. `requirements-lock.txt` records the installed versions; use a PyTorch build suitable for the machine's CPU/CUDA setup.
