"""LIME and SHAP word attributions for any model with ``predict_proba``.

Both methods treat the model as a black box and explain P(fake):

* LIME removes random subsets of words, then fits a weighted linear model to the
  changed predictions. A word's weight is how much P(fake) rises (+) or falls (-)
  when that word is present.
* SHAP (partition explainer with a text masker) gives each word its Shapley value:
  its average contribution to P(fake) over coalitions of words. The values add up
  to P(fake) minus the prediction with every word masked.

Positive weight = evidence for "fake", negative = evidence for "real".
These functions return plain dicts so the Streamlit app can reuse them directly.
"""

import html
import re

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)


def truncate_words(text, max_words=256):
    """Keep the first ``max_words`` whitespace tokens.

    Long articles make LIME/SHAP slow, and the BiLSTM/transformers only read the
    first 512 tokens anyway, so attributions are computed on the head of the text.
    """
    parts = text.split()
    return text if len(parts) <= max_words else " ".join(parts[:max_words])


def _merge(pairs):
    """Sum attributions per lower-cased word."""
    merged = {}
    for word, value in pairs:
        key = word.strip().lower()  # SHAP tokens keep their trailing separator
        if WORD.fullmatch(key):
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def lime_explain(model, text, num_samples=500, seed=0, max_words=256):
    from lime.lime_text import LimeTextExplainer

    text = truncate_words(text, max_words)
    explainer = LimeTextExplainer(class_names=["real", "fake"], split_expression=r"\W+",
                                  bow=True, random_state=seed)
    n_words = len(set(WORD.findall(text))) or 1
    exp = explainer.explain_instance(text, model.predict_proba, labels=(1,),
                                     num_features=n_words, num_samples=num_samples)
    return {
        "method": "lime",
        "text": text,
        "p_fake": float(model.p_fake([text])[0]),
        "weights": _merge(exp.as_list(label=1)),
        "intercept": float(exp.intercept[1]),
        "local_r2": float(exp.score),  # how well LIME's linear surrogate fits the model locally
    }


_SHAP_CACHE = {}


def shap_explain(model, text, max_evals=500, max_words=256):
    import shap

    text = truncate_words(text, max_words)
    if model.name not in _SHAP_CACHE:
        # Masked words become "..." (shap's default, runs collapsed): close to deleting them.
        masker = shap.maskers.Text(tokenizer=r"\W+")
        _SHAP_CACHE[model.name] = shap.Explainer(lambda xs: model.p_fake(list(xs)), masker, silent=True)
    explanation = _SHAP_CACHE[model.name]([text], max_evals=max_evals)
    tokens, values = explanation.data[0], explanation.values[0]
    return {
        "method": "shap",
        "text": text,
        "p_fake": float(model.p_fake([text])[0]),
        "weights": _merge(zip(tokens, values)),
        "base_value": float(np.ravel(explanation.base_values)[0]),
        "tokens": [str(t) for t in tokens],
        "token_values": [float(v) for v in values],
    }


def explain(model, text, method="lime", **kwargs):
    return (lime_explain if method == "lime" else shap_explain)(model, text, **kwargs)


def top_words(weights, k=10):
    """Top-k words by absolute weight, as (word, weight) pairs."""
    return sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:k]


def deletion_check(model, text, weights, k=5, seed=0):
    """Faithfulness test: does deleting the top-k words change the prediction more
    than deleting k random words?

    Words are taken in the direction of the predicted class (fake cues if the model
    says fake, real cues otherwise). Returns the drop in predicted-class probability
    for the explained words and for random words.
    """
    p = float(model.p_fake([text])[0])
    sign = 1 if p >= 0.5 else -1
    supporting = [w for w, v in sorted(weights.items(), key=lambda kv: -sign * kv[1]) if sign * v > 0][:k]
    vocab = sorted({w.lower() for w in WORD.findall(text)})
    if not supporting or len(vocab) <= k:
        return None
    rng = np.random.default_rng(seed)
    random_words = list(rng.choice(vocab, size=len(supporting), replace=False))

    def remove(words):
        drop = set(words)
        return WORD.sub(lambda m: "" if m.group(0).lower() in drop else m.group(0), text)

    probs = model.p_fake([remove(supporting), remove(random_words)])
    to_class = (lambda q: q) if sign > 0 else (lambda q: 1 - q)
    base = to_class(p)
    return {"explained_drop": float(base - to_class(probs[0])), "random_drop": float(base - to_class(probs[1])),
            "k": len(supporting)}


def highlight_html(text, weights, max_abs=None):
    """Colour each word by its weight: orange = pushes to fake, blue = pushes to real."""
    max_abs = max_abs or max([abs(v) for v in weights.values()] + [1e-9])
    pieces, last = [], 0
    for match in WORD.finditer(text):
        pieces.append(html.escape(text[last:match.start()]))
        word = match.group(0)
        value = weights.get(word.lower(), 0.0)
        alpha = min(abs(value) / max_abs, 1.0)
        colour = "230,120,30" if value > 0 else "40,110,220"
        title = f"{value:+.4f}"
        pieces.append(f'<span title="{title}" style="background:rgba({colour},{alpha:.2f});'
                      f'border-radius:3px;padding:0 1px">{html.escape(word)}</span>')
        last = match.end()
    pieces.append(html.escape(text[last:]))
    return "".join(pieces)
