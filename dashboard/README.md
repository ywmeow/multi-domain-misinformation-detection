# Signal dashboard

A Streamlit app with a hero and a workspace for comparing the saved BiLSTM,
**Jruvika GraphSAGE** and fine-tuned **RoBERTa + Muon** classifiers on the same
headline, claim, or article. **Explain predictions** highlights the words that pushed
each model towards fake or real, using LIME.

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
All three classifiers run on CPU. No training datasets, original GloVe vectors,
Hugging Face account, or GPU are required. The app itself never downloads anything:
RoBERTa is loaded with `local_files_only=True`.

## Behavior

- Paste up to 20,000 characters and select **Analyze text**. Ctrl/Cmd+Enter also submits.
- Example prompts fill the input without assuming or displaying a ground-truth label.
- Both models receive the same NFC-normalized text with collapsed whitespace,
  then apply their own saved preprocessing.
- The BiLSTM uses the first 512 spaCy tokens and a 0.5 fake-class threshold.
  A truncation notice appears if the input exceeds that token limit.
- RoBERTa uses the first 512 subword tokens and a 0.5 threshold (label 1 = fake, as in
  training). Its score matches the usage example in Trung's Hugging Face README.
- GraphSAGE uses the trusted local Jruvika bundle, including its fitted TF-IDF,
  SVD, neighbors, anchors, neural weights, and saved decision threshold (currently 0.55).
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
- LIME needs hundreds of model calls, so it runs only on request. BiLSTM and GraphSAGE
  use 300 perturbations of the first 200 words (under 2 s). RoBERTa uses 100
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
- Checkpoint failures are isolated, so the other model can still return a result.

The app does not write submitted text to disk or use browser storage. Text is
sent to the Streamlit server and retained in the active session's memory.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit page setup, session state, and request handling. |
| `inference.py` | Cached CPU model loaders (BiLSTM, GraphSAGE, RoBERTa) and a common prediction result. |
| `explain.py` | On-demand LIME explanations for every model that produced a prediction. |
| `download_transformer.py` | One-time download of the pinned RoBERTa + Muon checkpoint. |
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
graph/Models/jruvika_graphsage.joblib
explainability/explainers.py
```

The GraphSAGE bundle was saved with scikit-learn 1.5.2; the original environment
is recorded in `graph/requirements-lock.txt`. Newer scikit-learn versions may emit
a compatibility warning. For reproducible deployment, use Python 3.11 or 3.12 and
the original compatible NumPy/SciPy/scikit-learn versions. Only load trusted model
artifacts. The dashboard has no checkpoint-upload endpoint.

The model cards name their different training sources: BiLSTM used all four
datasets, while GraphSAGE used Jruvika. Scores are not calibrated probabilities
of truth, and headline-only accuracy has not been established by the project's
mixed claim/article evaluation.

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
