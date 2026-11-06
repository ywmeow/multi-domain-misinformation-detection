"""Download the fine-tuned RoBERTa (Muon) checkpoint once, for the dashboard.

Run from the repository root:

    python dashboard/download_transformer.py

Fetches only ``models/roberta-base-muon/`` (about 500 MB) from Trung's public
Hugging Face repository, pinned to the commit the dashboard was tested with, into the
standard Hugging Face cache. The dashboard then loads it with ``local_files_only``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.inference import (  # noqa: E402
    TRANSFORMER_REPO,
    TRANSFORMER_REVISION,
    TRANSFORMER_SUBFOLDER,
)


def main() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        TRANSFORMER_REPO,
        revision=TRANSFORMER_REVISION,
        allow_patterns=[f"{TRANSFORMER_SUBFOLDER}/*"],
    )
    print(f"RoBERTa (Muon) checkpoint ready in {Path(path) / TRANSFORMER_SUBFOLDER}")


if __name__ == "__main__":
    main()
