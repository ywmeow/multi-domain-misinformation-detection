"""TF-IDF (word 1-2gram, fit on train only) + logistic regression reference on the pooled split."""
import json, pathlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

R = pathlib.Path("/scratch/users/astar/ares/suns1/trung/nlp")
tr, te = pd.read_parquet(R / "data/pooled_train.parquet"), pd.read_parquet(R / "data/pooled_test.parquet")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=500_000, sublinear_tf=True)
clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced")
clf.fit(vec.fit_transform(tr.text), tr.label_id)
pred = clf.predict(vec.transform(te.text))
res = {"overall": round(f1_score(te.label_id, pred, average="macro"), 4)}
for k, g in te.assign(p=pred).groupby("dataset"):
    res[k] = round(f1_score(g.label_id, g.p, average="macro"), 4)
(R / "results/tfidf_lr").mkdir(exist_ok=True)
json.dump(res, open(R / "results/tfidf_lr/metrics.json", "w"), indent=1); print(res)
