"""Streamlit Community Cloud entrypoint; prepares RoBERTa before opening the UI."""

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import main as dashboard_main
from dashboard.download_transformer import main as download_transformer


@st.cache_resource(show_spinner="Preparing models for first use. This may take a few minutes...")
def prepare_models() -> None:
    # Reuse the pinned public checkpoint and the inference loader's default cache.
    # A failed download is not cached, so a subsequent attempt can retry it.
    download_transformer()


def main() -> None:
    prepare_models()
    dashboard_main()


if __name__ == "__main__":
    main()
