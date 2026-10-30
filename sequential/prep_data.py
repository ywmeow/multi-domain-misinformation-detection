"""Pool existing binary splits, following trung/prep_data.py."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root.parent)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--extracted", type=Path, default=root / "data/extracted")
    args = parser.parse_args()

    sources = {
        "amt": args.repo / "preprocessing/output/amt",
        "multifc": args.repo / "preprocessing/output/multifc",
        "mc_fake": args.extracted / "mc_fake",
        "jruvika": args.extracted / "jruvika",
    }
    columns = ["id", "text", "label_id", "domain", "group_id"]
    args.data.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        frames = []
        for name, directory in sources.items():
            with (directory / f"binary_{split}.jsonl").open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            frame = pd.DataFrame(rows)[columns]
            frame["dataset"] = name
            frames.append(frame)

        pooled = pd.concat(frames, ignore_index=True)
        pooled["label_id"] = pooled["label_id"].astype(int)
        pooled["split"] = split
        pooled.to_parquet(args.data / f"pooled_{split}.parquet", index=False)
        print(split, len(pooled))
        print(pooled.groupby("dataset").label_id.agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()
