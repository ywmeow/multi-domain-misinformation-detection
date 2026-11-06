"""Load the team's binary train/dev/test exports for all four datasets.

AMT and MultiFC are read straight from ``preprocessing/output``. MC-Fake and
Jruvika are stored as 7z archives there; they are extracted once into
``explainability/cache/extracted`` (ignored by Git) with py7zr, so 7-Zip does
not need to be installed.
"""

from functools import lru_cache
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "preprocessing" / "output"
CACHE = Path(__file__).resolve().parent / "cache"
EXTRACTED = CACHE / "extracted"

DATASETS = ("amt", "multifc", "mc_fake", "jruvika")
DISPLAY = {"amt": "AMT", "multifc": "MultiFC", "mc_fake": "MC-Fake", "jruvika": "Jruvika"}
LABELS = ("real", "fake")  # label_id 0 = real, 1 = fake
COLUMNS = ["id", "text", "label_id", "dataset", "domain"]


def _extract(dataset):
    """Extract ``<dataset>/binary_*.jsonl`` from the archive if not done yet."""
    target = EXTRACTED / dataset
    if all((target / f"binary_{s}.jsonl").exists() for s in ("train", "dev", "test")):
        return target
    import py7zr

    EXTRACTED.mkdir(parents=True, exist_ok=True)
    if dataset == "mc_fake":
        # Multi-volume archive: join the volumes into one file first.
        archive = CACHE / "mc_fake_joined.7z"
        if not archive.exists():
            with open(archive, "wb") as joined:
                for part in sorted((OUTPUT / "mc_fake").glob("mc_fake.7z.0*")):
                    with open(part, "rb") as stream:
                        shutil.copyfileobj(stream, joined)
    else:
        archive = OUTPUT / "jruvika" / "jruvika.7z"
    with py7zr.SevenZipFile(archive, "r") as zf:
        wanted = [n for n in zf.getnames() if n.startswith(f"{dataset}/binary_") and n.endswith(".jsonl")]
        zf.extract(path=EXTRACTED, targets=wanted)
    return target


@lru_cache(maxsize=None)
def load_split(dataset, split):
    """Return one dataset split as a DataFrame with the standard columns."""
    if dataset not in DATASETS or split not in ("train", "dev", "test"):
        raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
    folder = OUTPUT / dataset if dataset in ("amt", "multifc") else _extract(dataset)
    frame = pd.read_json(folder / f"binary_{split}.jsonl", lines=True)
    frame["dataset"] = dataset
    if "domain" not in frame:
        frame["domain"] = dataset
    frame["label_id"] = frame["label_id"].astype(int)
    return frame[COLUMNS].copy()


def load_pooled(split, datasets=DATASETS):
    return pd.concat([load_split(d, split) for d in datasets], ignore_index=True)


def sample_split(dataset, split="test", n=20, seed=42):
    """Label-balanced random sample (n/2 real, n/2 fake where available)."""
    frame = load_split(dataset, split)
    parts = []
    for label in (0, 1):
        group = frame[frame.label_id == label]
        parts.append(group.sample(min(len(group), n // 2), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
