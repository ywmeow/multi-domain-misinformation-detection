"""Load GloVe text vectors for the existing training-only vocabulary."""

import torch


def load_glove(path, vocab, embedding_dim):
    ordinary = {word: index for word, index in vocab.items() if word not in ("<PAD>", "<UNK>")}
    needed = set(ordinary) | {word.lower() for word in ordinary}
    vectors = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            word, _, values = line.rstrip().partition(" ")
            if word not in needed:
                continue
            vector = torch.tensor([float(value) for value in values.split()])
            if vector.numel() != embedding_dim or not torch.isfinite(vector).all():
                raise ValueError(f"Invalid GloVe vector for {word!r}; expected {embedding_dim} finite values")
            vectors[word] = vector
    if not vectors:
        raise ValueError("No vocabulary tokens matched the GloVe file")

    weights = torch.empty(len(vocab), embedding_dim).normal_(mean=0, std=0.05)
    exact, lowercase = 0, 0
    for word, index in ordinary.items():
        if word in vectors:
            weights[index] = vectors[word]
            exact += 1
        elif word.lower() in vectors:
            weights[index] = vectors[word.lower()]
            lowercase += 1
    weights[vocab["<PAD>"]].zero_()
    coverage = {
        "source": str(path.resolve()),
        "embedding_dim": embedding_dim,
        "ordinary_tokens": len(ordinary),
        "exact_matches": exact,
        "lowercase_matches": lowercase,
        "oov_tokens": len(ordinary) - exact - lowercase,
        "vocab_coverage": (exact + lowercase) / len(ordinary),
    }
    return weights, coverage
