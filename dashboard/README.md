# Signal dashboard

A Streamlit app with a hero and a workspace for comparing the saved BiLSTM and
**Jruvika GraphSAGE** classifiers on the same headline, claim, or article.
The RoBERTa + Muon panel stays unavailable until its weights are added.

## Run locally

From the repository root, using Python 3.10+:

```powershell
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

For the project's existing Windows environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
CPU is used for both classifiers. No training datasets, original GloVe vectors,
Hugging Face account, or GPU are required.

## Behavior

- Paste up to 20,000 characters and select **Analyze text**. Ctrl/Cmd+Enter also submits.
- Example prompts fill the input without assuming or displaying a ground-truth label.
- Both models receive the same NFC-normalized text with collapsed whitespace,
  then apply their own saved preprocessing.
- The BiLSTM uses the first 512 spaCy tokens and a 0.5 fake-class threshold.
  A truncation notice appears if the input exceeds that token limit.
- GraphSAGE uses the trusted local Jruvika bundle, including its fitted TF-IDF,
  SVD, neighbors, anchors, neural weights, and saved decision threshold (currently 0.55).
- Models are loaded lazily and cached in the server process. A lock protects
  shared model instances during inference across multiple sessions.
- Each result includes the predicted class, uncalibrated fake-class score,
  and prediction time. Total time also includes initial model loading.
- Results identify the last submitted text; editing the input marks existing
  results as out of date. Clear resets both the input and results.
- Download results exports JSON, including unavailable/error states. No ensemble
  verdict is generated, and the absent transformer is never included in agreement.
- Checkpoint failures are isolated, so the other model can still return a result.

The app does not write submitted text to disk or use browser storage. Text is
sent to the Streamlit server and retained in the active session's memory.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit page setup, session state, and request handling. |
| `inference.py` | Cached CPU model loaders and a common prediction result. |
| `frontend/index.html` | The hero, input form, and model panels. |
| `frontend/styles.css` | Component styles, grouped by section and responsive layout. |
| `frontend/streamlit.css` | Styles for the surrounding Streamlit page and the font import. |
| `frontend/app.js` | Input events, result rendering, and JSON downloads. |

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

## Adding the transformer later

Add a local-only inference adapter for the fine-tuned `roberta_muon_lr5e-5`
checkpoint, its matching tokenizer and configuration, and then enable the third
panel. `analyze()` returns a common prediction structure with model ID, status,
class, fake-class score, threshold and time. No transformer dependency or
download is included at this stage.
