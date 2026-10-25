"""Build one pooled multi-domain parquet from the team's four binary exports.

Reads <repo>/preprocessing/output/{amt,multifc} and the extracted mc_fake / jruvika
archives (see README). Output: <data>/pooled_{train,dev,test}.parquet with columns
id, text, label_id, dataset, domain, split.
"""
import argparse, json, pathlib
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--repo", default="/scratch/users/astar/ares/suns1/trung/nlp/repo")
p.add_argument("--data", default="/scratch/users/astar/ares/suns1/trung/nlp/data")
a = p.parse_args()
repo, data = pathlib.Path(a.repo), pathlib.Path(a.data)
src = {
    "amt": repo / "preprocessing/output/amt",
    "multifc": repo / "preprocessing/output/multifc",
    "mc_fake": data / "mc_fake/mc_fake",
    "jruvika": data / "jruvika/jruvika",
}
cols = ["id", "text", "label_id", "domain", "group_id"]
for split in ["train", "dev", "test"]:
    frames = []
    for name, d in src.items():
        rows = [json.loads(l) for l in open(d / f"binary_{split}.jsonl", encoding="utf-8")]
        df = pd.DataFrame(rows)[cols]
        df["dataset"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["label_id"] = out["label_id"].astype(int)
    out["split"] = split
    out.to_parquet(data / f"pooled_{split}.parquet")
    print(split, len(out))
    print(out.groupby("dataset").label_id.agg(["count", "mean"]).round(3))
