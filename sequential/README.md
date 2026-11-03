# BiLSTM misinformation classifier

Binary misinformation classification (`0=real`, `1=fake`) on the pooled AMT,
MultiFC, MC-Fake, and Jruvika datasets. The pipeline uses the team's binary
exports and preserves their train/dev/test split membership.

| File | Responsibility |
|---|---|
| `prep_data.py` | Pool the four datasets into train/dev/test Parquet files. |
| `data.py` | Build the training-only vocabulary, tokenize splits, and pad batches. |
| `embeddings.py` | Load GloVe vectors for the vocabulary and report coverage. |
| `model.py` | Define the BiLSTM and its forward pass. |
| `train.py` | Train, validate, select a checkpoint, and evaluate test. |
| `predict.py` | Load the distributed checkpoint and classify new text. |

## Setup

Requires Python 3.10+. Training-data preparation also requires 7-Zip for
extracting the dataset archives. Run all commands from the repository root.

```sh
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Install dependencies:

```sh
python -m pip install -r sequential/requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).
The training script automatically selects CUDA when available, otherwise CPU.

## Predict with the saved model

The selected GloVe 100d BiLSTM run is included in
`sequential/checkpoints/bilstm_glove100_seed42/`. After installing the
dependencies, run:

```sh
python sequential/predict.py --text "The council approved a new public transport plan."
```

The command returns JSON with `label_id` (`0=real`, `1=fake`), `label`, and
`p_fake`. It uses CPU by default; `--device cuda` enables a compatible GPU.
Use `--checkpoint path/to/run` to load another saved run directory.

Prediction needs `best.pt`, its matching `vocab.json`, and
`tokenizer_settings.json`, together with `model.py`, `predict.py`, PyTorch,
and spaCy. The distributed run also includes its configuration, history,
metrics, and dev/test predictions for review. Original Colab paths in the
configuration record the training environment; they do not need to exist locally.

No training-data extraction, token cache, original GloVe download, or retraining
is required for prediction. The checkpoint already contains the fine-tuned
embedding table. New input receives NFC Unicode normalization and whitespace
collapse, then the same spaCy tokenizer and saved vocabulary, with head
truncation at 512 tokens. Unknown words map to `<UNK>`; the vocabulary and
embeddings stay fixed. Keep each checkpoint's vocabulary and tokenizer settings
with that checkpoint, even if you regenerate a training cache later.

These are classifier scores, not independent fact verification. Headline-only
input is supported, but the recorded evaluation includes claims and full
articles; it does not establish accuracy on a new headline-only collection.

## What is shared through Git

| Path under `sequential/` | Git policy | Purpose |
|---|---|---|
| `data/` | Ignored | Extracted JSONL, pooled data, and optional source GloVe vectors; recreate for training. |
| `cache/` | Ignored | Training vocabulary and tokenized splits; regenerate with `data.py`. |
| `checkpoints/bilstm_glove100_seed42/` | Included | The selected trained model and its matching inference files and results. |
| Other directories under `checkpoints/` | Ignored | New experiments; share a selected run explicitly when needed. |
| `notebook/`, `PLAN.md` | Ignored | Local notebooks, bundles, and planning notes. |

Cloning the repository provides the selected model and source code. Teammates
who only need predictions can stop after setup and the prediction command.
For training, recreate `data/` and `cache/` using the steps below, or obtain
matching generated files separately. Once the token cache exists, `train.py`
can train directly from it without the pooled or extracted data. GloVe training
additionally needs the original vector file to initialize a fresh model.

## Prepare the data

Use the existing `binary_{train,dev,test}.jsonl` exports from the preprocessing
pipeline. AMT and MultiFC are read directly from `preprocessing/output/`.
MC-Fake and Jruvika are distributed as archives in that directory; extract
them once only if the binary JSONL files are not already available.

With `7z` available on the system PATH, extract to the default input directory:

```sh
7z x preprocessing/output/mc_fake/mc_fake.7z.001 -osequential/data/extracted "mc_fake/binary_*.jsonl"
7z x preprocessing/output/jruvika/jruvika.7z -osequential/data/extracted "jruvika/binary_*.jsonl"
```

Then pool the exports:

```sh
python sequential/prep_data.py
```

If MC-Fake and Jruvika have already been extracted under
`preprocessing/output/mc_fake/` and `preprocessing/output/jruvika/`, reuse them:

```sh
python sequential/prep_data.py --extracted preprocessing/output
```

For another extraction location, set `--extracted` to the parent directory
containing both `mc_fake/` and `jruvika/`. Existing pooled Parquet files can be
used directly for tokenization; existing token caches can be used directly for
training. These generated files are ignored by Git and must be shared
separately when reusing them across machines.

Corresponding splits are concatenated in the order AMT, MultiFC, MC-Fake,
Jruvika, preserving source row order, text, labels, and split membership.

| Dataset | Train | Dev | Test |
|---|---:|---:|---:|
| AMT | 384 | 48 | 48 |
| MultiFC | 19,610 | 2,445 | 2,434 |
| MC-Fake | 22,668 | 2,835 | 2,832 |
| Jruvika | 3,209 | 399 | 401 |
| **Pooled** | **45,871** | **5,727** | **5,715** |

Outputs are `sequential/data/pooled_{train,dev,test}.parquet`, with columns
`id, text, label_id, domain, group_id, dataset, split`. Pooling also prints
per-dataset row counts and fake-class proportions.

Path overrides: `--repo`, `--extracted`, and `--data`. See the
[preprocessing guide](../preprocessing/README.md) for source dataset details.

## Tokenization and vocabulary

```sh
python sequential/data.py
```

Tokenization uses `spacy.blank("en")`, preserving case, punctuation, numbers,
and negation. No language-model download is required. The vocabulary is built
only from training text, with minimum frequency 2 and up to 50,000 ordinary
tokens, plus `<PAD>=0` and `<UNK>=1`. Tied frequencies preserve first-seen order.
Rare or unseen tokens map to UNK; the vocabulary stays fixed for dev/test.

Sequences retain their first 512 tokens. Empty text becomes one UNK token.
`collate_batch` pads to the longest sequence in each batch and returns integer
token IDs, true sequence lengths, and labels.

Files written to `sequential/cache/`:

- `vocab.json`: token-to-ID mapping.
- `settings.json`: tokenizer settings and package versions.
- `tokenized_{train,dev,test}.parquet`: source columns plus `token_length`
  before truncation and `input_ids` after truncation.
- `diagnostics.json`: per-dataset train/dev length statistics, truncation
  rates, and unknown-token rates.

Options: `--data`, `--cache`, `--min-freq`, `--max-vocab`, and `--max-length`.
The vocabulary limit excludes PAD and UNK. Preparation outputs are replaced
on reruns.

## Model and training

The model consists of embeddings, embedding dropout, a one-layer bidirectional
LSTM, masked mean pooling, classifier dropout, and a linear layer producing
two logits. Packed sequences exclude padding from both recurrent directions;
mean pooling excludes padding, and the PAD embedding stays zero. Only text
token IDs and lengths enter the model; dataset metadata is retained for reports.

Download the GloVe 6B archive from the
[Stanford GloVe project](https://nlp.stanford.edu/projects/glove/) and extract
`glove.6B.100d.txt` to `sequential/data/embeddings/`. Then run:

```sh
python sequential/train.py --glove sequential/data/embeddings/glove.6B.100d.txt --embedding-dim 100 --epochs 10 --output sequential/checkpoints/bilstm_glove100_retrain
```

This command uses the distributed run's 10-epoch cap. Omitting `--epochs`
uses the current trainer's default of 20; each attempt needs a new output
directory. The distributed checkpoint remains in its separate run directory.

| Setting | Value |
|---|---|
| Embeddings | GloVe 100d; dimension must match the vector file |
| Hidden size | 128 per direction |
| Embedding / classifier dropout | 0.3 / 0.3 |
| Batch size / seed | 32 / 42 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `0.01` |
| Embedding warmup | Frozen for epoch 1, then fine-tuned at `1e-4` |
| Loss / gradient clipping | Unweighted cross-entropy / norm 1.0 |
| Default epoch limit / early-stopping patience | 20 / 3; distributed run capped at 10 |

GloVe lookup prefers exact spelling, then lowercase fallback. Missing tokens
and UNK receive small random vectors. Coverage of ordinary vocabulary entries
is printed and saved in `config.json`. All embedding rows are frozen during
warmup, including unmatched entries. Patience resets when embeddings unfreeze.

For randomly initialized embeddings, omit `--glove`; the embedding dimension
defaults to 128 and embeddings train immediately at the main learning rate.
`--embedding-dropout 0` and `--patience 0` disable embedding dropout and early
stopping respectively. Other options include `--epochs`, `--batch-size`,
`--hidden-size`, `--lr`, `--device`, `--threads`, and `--cache`; use `--help`
for the full list. Each run requires a new output directory.

## Validation and test evaluation

Training examples are shuffled each epoch. After each epoch, macro-F1 is
computed separately for the four dev datasets. The checkpoint selection score
is the unweighted mean of these four scores, giving each dataset 25% weight.
Training stops after three consecutive epochs without a new best score,
following warmup, or at the epoch limit.

`train.py` reloads that run's best checkpoint and evaluates test once with
dropout disabled and no gradient updates. Predictions use `p_fake >= 0.5`.
Reports include accuracy, macro-F1, real/fake F1, cross-entropy, and AUROC
where both classes exist. Domain reports include dataset/domain groups with
at least 10 examples.

Pooled macro-F1 is computed from all predictions combined. Mean dataset
macro-F1 averages the four separately computed dataset scores. These are
different aggregations; the latter gives equal weight to datasets. Mean test
dataset macro-F1 can be calculated from the four dataset entries in the report.

## Run outputs

The selected output directory contains:

- `best.pt`: model weights and constructor settings, including all embeddings.
- `vocab.json`, `tokenizer_settings.json`, `config.json`: vocabulary, tokenizer,
  training settings, versions, and GloVe coverage when applicable.
- `history.json`: epoch-average training loss, dev cross-entropy, dev scores,
  embedding learning rate, and freeze status.
- `metrics.json`: selected/completed epochs, final/selected training losses,
  dev/test reports, timing, parameter count, and hardware/memory information.
- `dev_preds.parquet`, `test_preds.parquet`: IDs, labels, dataset/domain/group
  metadata, token lengths, and `p_fake`, preserving source order.

Training loss is averaged during updates with dropout active. Dev/test
cross-entropy is computed from predicted probabilities in evaluation mode.

For inference, load `best.pt` with `torch.load(..., map_location="cpu",
weights_only=True)`, construct `BiLSTM(**checkpoint["model_config"])`, load
`checkpoint["model"]` with `load_state_dict`, and call `eval()`. The source
GloVe file is unnecessary for inference. Optimizer state is not saved for
training resumption. Generated data, caches, and new checkpoint directories are
ignored by Git. The selected `bilstm_glove100_seed42` run is explicitly included
so teammates receive the trained weights and matching inference files.

## Recorded results

| Model | Pooled test macro-F1 | AMT | MultiFC | MC-Fake | Jruvika |
|---|---:|---:|---:|---:|---:|
| Distributed BiLSTM, GloVe 100d, selected epoch 10 | 0.8103 | 0.4535 | 0.6589 | 0.7695 | 0.9374 |
| TF-IDF + logistic regression | 0.8144 | 0.5781 | 0.6463 | 0.7577 | 0.9450 |

These GloVe results come from the distributed run's
[metrics.json](checkpoints/bilstm_glove100_seed42/metrics.json), with its
configuration in [config.json](checkpoints/bilstm_glove100_seed42/config.json).
It completed all 10 configured epochs and selected epoch 10 with dev mean
dataset macro-F1 **0.7468**. Its test accuracy was **0.8163**, mean dataset
macro-F1 **0.7048**, cross-entropy **0.3761**, and AUROC **0.9077**. GloVe matched
**94.51%** of the 50,000 ordinary vocabulary entries. Training took approximately
**10.1 minutes on a Tesla T4**. These are single-seed results, and AMT has only
48 test examples.

The reference baseline is recorded in
[Trung's baseline metrics](../trung/results/tfidf_lr/metrics.json).
Transformer reference results and their training settings are documented in
[trung/README.md](../trung/README.md).

## Checks

```sh
python -m unittest discover -s sequential/tests -v
```

The small CPU tests cover GloVe matching and validation, freezing/unfreezing,
padding invariance, early stopping, checkpoint reload, prediction alignment,
and prediction using only a saved model, vocabulary, and tokenizer settings.
