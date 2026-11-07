# Signal dashboard

A Streamlit app with a hero and a workspace for comparing one model from each of the
project's four families on the same headline, claim, or article: the **TF-IDF + logistic
regression** baseline, the **BiLSTM**, **GraphSAGE** and the fine-tuned **RoBERTa + Muon**.
All four were trained on the same pooled training splits of AMT, MultiFC, MC-Fake and
Jruvika, so the cards compare like with like. **Explain predictions** highlights the
words that pushed each model towards fake or real, using LIME.

| Card | Model | Pooled test macro-F1 |
| --- | --- | ---: |
| Traditional | TF-IDF (word 1-2 grams) + logistic regression | 0.814 |
| Sequential | GloVe BiLSTM | 0.810 |
| Graph | GraphSAGE on a TF-IDF/SVD document graph | 0.796 |
| Transformer | roberta-base fine-tuned with Muon | 0.839 |

## Run locally

From the repository root, using Python 3.10+:

```powershell
python -m pip install -r dashboard/requirements.txt
python dashboard/download_transformer.py
python -m streamlit run dashboard/app.py
```

`download_transformer.py` is needed once per machine. It fetches Trung's fine-tuned
`models/roberta-base-muon` (about 500 MB) from the public Hugging Face repository
`ngqtrung/ee6405-misinfo-detection`, pinned to commit `7837df8`, into the standard
Hugging Face cache. Without it the app still runs; the RoBERTa card shows
"Not connected" with the command to run.

For the project's existing Windows environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
All four classifiers run on CPU. No training datasets, original GloVe vectors,
Hugging Face account, or GPU are required. The app itself never downloads anything:
RoBERTa is loaded with `local_files_only=True`.

## Behavior

- Paste up to 20,000 characters and select **Analyze text**. Ctrl/Cmd+Enter also submits.
- Example prompts fill the input without assuming or displaying a ground-truth label.
- Every model receives the same NFC-normalized text with collapsed whitespace,
  then applies its own saved preprocessing.
- TF-IDF + logistic regression reads the whole text and uses a 0.5 threshold.
- The BiLSTM uses the first 512 spaCy tokens and a 0.5 fake-class threshold.
  A truncation notice appears if the input exceeds that token limit.
- RoBERTa uses the first 512 subword tokens and a 0.5 threshold (label 1 = fake, as in
  training). Its score matches the usage example in Trung's Hugging Face README.
- GraphSAGE uses Swetha's pooled bundle (`graph/Models/Model_pooled.joblib`), including
  its fitted TF-IDF, SVD, neighbors, anchors, neural weights, and the dev-selected
  neighbor count (5) and decision threshold (0.53). Its score matches
  `graph/Model_pooled/predict.py`.
- Models are loaded lazily and cached in the server process. A lock protects
  shared model instances during inference across multiple sessions.
- Each result includes the predicted class, uncalibrated fake-class score,
  and prediction time. Total time also includes initial model loading.
- Results identify the last submitted text; editing the input marks existing
  results as out of date. Clear resets both the input and results.
- The status line reports whether all completed models agree, or how many say fake.
  No ensemble verdict is generated, and a model that did not run is never counted.
- **Explain predictions** runs LIME on the analyzed text for every model that
  produced a result. LIME deletes random subsets of words, re-scores each variant and
  fits a local linear model; each word's weight is how much it raised (orange, towards
  fake) or lowered (lime, towards real) that model's fake-class score. Each card then
  shows the highlighted text and its strongest words in each direction. Hover a word to
  see its weight.
- LIME needs hundreds of model calls, so it runs only on request. TF-IDF uses 500
  perturbations of the first 256 words; BiLSTM and GraphSAGE use 300 perturbations of
  the first 200 words (each under 2 s). RoBERTa uses 100
  perturbations of the first 100 words and every CPU core while it runs, about
  5-15 s on a 12-thread laptop. Results are seeded, so the same text gives the same
  explanation.
- When a model is so confident that deleting any word moves its score by less than one
  point, the card says so instead of showing near-zero weights as evidence.
- Explanations reuse `lime_explain` from `explainability/explainers.py`, so the app and
  the project's explainability analysis compute them the same way. They show what a
  model relies on, not whether the text is true.
- Download results exports JSON, including unavailable/error states and the current
  explanation.
- Checkpoint failures are isolated, so the other models can still return results.

The app does not write submitted text to disk or use browser storage. Text is
sent to the Streamlit server and retained in the active session's memory.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit page setup, session state, and request handling. |
| `inference.py` | Cached CPU model loaders (TF-IDF, BiLSTM, GraphSAGE, RoBERTa) and a common prediction result. |
| `explain.py` | On-demand LIME explanations for every model that produced a prediction. |
| `download_transformer.py` | One-time download of the pinned RoBERTa + Muon checkpoint. |
| `train_tfidf.py` | Trains and saves the TF-IDF + logistic regression baseline (about a minute). |
| `models/tfidf_lr.joblib` | The saved TF-IDF baseline, with its settings and test scores. |
| `frontend/index.html` | The hero, input form, and model panels. |
| `frontend/styles.css` | Component styles, grouped by section and responsive layout. |
| `frontend/streamlit.css` | Styles for the surrounding Streamlit page and the font import. |
| `frontend/app.js` | Input events, result rendering, word highlighting, and JSON downloads. |

The interface uses Streamlit's v2 component API without a separate frontend
build. Animations use CSS and respect reduced-motion preferences. Geist fonts
and decorative Picsum images load externally; system fonts and CSS backgrounds
provide fallbacks. Model inference is entirely local.

## Checkpoints and compatibility

Keep these repository-provided files in place:

```text
sequential/checkpoints/bilstm_glove100_seed42/
  best.pt
  vocab.json
  tokenizer_settings.json
graph/Models/Model_pooled.joblib
graph/model.py
dashboard/models/tfidf_lr.joblib
explainability/explainers.py
```

The GraphSAGE and TF-IDF bundles were saved with scikit-learn 1.5.2; the original environment
is recorded in `graph/requirements-lock.txt`. Newer scikit-learn versions may emit
a compatibility warning. For reproducible deployment, use Python 3.11 or 3.12 and
the original compatible NumPy/SciPy/scikit-learn versions. Only load trusted model
artifacts. The dashboard has no checkpoint-upload endpoint.

Every card uses a model trained on the same pooled training splits, so differences
between cards come from the model family, not the training data. Scores are not
calibrated probabilities of truth, and headline-only accuracy has not been established
by the project's mixed claim/article evaluation.

## TF-IDF baseline

`dashboard/models/tfidf_lr.joblib` (10 MB) is committed, so the card works after a
pull. To rebuild it, run `python dashboard/train_tfidf.py`. It uses the settings of
`trung/baseline_tfidf.py` (word 1-2 grams, min_df 3, up to 500k features, sublinear tf;
logistic regression with C=10 and balanced class weights) on the pooled training
splits, loads the data with `explainability/data.py`, and reproduces Trung's recorded
0.8144 test macro-F1 (0.8143).

## Checks

```powershell
python -m unittest discover -s dashboard/tests -v
```

## Transformer checkpoint

The RoBERTa card uses `models/roberta-base-muon`: roberta-base fine-tuned on the
pooled four datasets with the Muon optimizer (0.839 overall test macro-F1, the best
run in `trung/README.md`). The weights are too large for Git, so they stay on
Hugging Face. `TRANSFORMER_REVISION` in `inference.py` pins the repository commit;
change it there and rerun `download_transformer.py` to use a newer upload. Its tests
in `tests/test_inference.py` are skipped until the weights are downloaded.
