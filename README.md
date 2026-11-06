# multi-domain-misinformation-detection
EE6405 Group 43 2026/S1
## Preprocessing

See the [preprocessing guide](preprocessing/README.md) for setup,
run commands, output descriptions and sanity checks.

## Sequential model

See the [BiLSTM guide](sequential/README.md) for prediction with the included
GloVe checkpoint, training commands, and recorded results. Generated sequential
data and token caches are ignored; the selected trained model is included.

## Explainability (LIME / SHAP)

See the [explainability guide](explainability/README.md) for LIME and SHAP
explanations of the TF-IDF, BiLSTM and GraphSAGE models, the shortcut audit,
the counterfactual stress test and the cross-domain transfer analysis.
