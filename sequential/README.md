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

## Setup

Requires Python 3.10+ and 7-Zip for extracting the dataset archives. Run all
commands from the repository root.

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
python sequential/train.py --glove sequential/data/embeddings/glove.6B.100d.txt --embedding-dim 100 --output sequential/checkpoints/bilstm_glove100
```

| Setting | Value |
|---|---|
| Embeddings | GloVe 100d; dimension must match the vector file |
| Hidden size | 128 per direction |
| Embedding / classifier dropout | 0.3 / 0.3 |
| Batch size / seed | 32 / 42 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `0.01` |
| Embedding warmup | Frozen for epoch 1, then fine-tuned at `1e-4` |
| Loss / gradient clipping | Unweighted cross-entropy / norm 1.0 |
| Epoch limit / early-stopping patience | 20 / 3 |

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
training resumption. Generated data, caches, and checkpoints are ignored by Git.

## Recorded results

| Model | Pooled test macro-F1 | AMT | MultiFC | MC-Fake | Jruvika |
|---|---:|---:|---:|---:|---:|
| BiLSTM, random embeddings, selected epoch 2 | 0.7914 | 0.5312 | 0.6040 | 0.7439 | 0.9127 |
| BiLSTM, GloVe 100d, selected epoch 12 | 0.8092 | 0.4844 | 0.6486 | 0.7743 | 0.9423 |
| TF-IDF + logistic regression | 0.8144 | 0.5781 | 0.6463 | 0.7577 | 0.9450 |

The GloVe run stopped after 15 epochs, selecting epoch 12 with dev mean dataset
macro-F1 **0.7529**. Its test accuracy was **0.8138**, mean dataset macro-F1
**0.7124**, cross-entropy **0.4092**, and AUROC **0.9051**. The random-embedding
reference used no embedding dropout. Transformer reference results and their
training settings are documented in [trung/README.md](../trung/README.md).

## Checks

```sh
python -m unittest discover -s sequential/tests -v
```

The small CPU tests cover GloVe matching and validation, freezing/unfreezing,
padding invariance, early stopping, checkpoint reload, and prediction alignment.
