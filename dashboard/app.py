"""Run with: python -m streamlit run dashboard/app.py"""

from pathlib import Path
import sys

import streamlit as st
from streamlit.components.v2 import component


# Streamlit accepts this entrypoint from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.inference import MAX_CHARACTERS, analyze


ASSETS = Path(__file__).resolve().parent / "frontend"


def read_asset(filename: str) -> str:
    return (ASSETS / filename).read_text(encoding="utf-8")


@st.cache_resource
def dashboard_component(html: str, css: str, javascript: str):
    """Register the component once for each version of the frontend assets."""
    return component(
        "signal_dashboard",
        html=html,
        css=css,
        js=javascript,
        isolate_styles=True,
    )


def process_request(request: dict) -> None:
    """Apply a submitted analysis or reset to the current Streamlit session."""
    state = st.session_state
    state.last_request_id = request["id"]
    state.input_error = None

    if request["action"] == "reset":
        state.analysis = None
        return

    try:
        state.analysis = analyze(request.get("text", ""))
    except ValueError as error:
        state.input_error = str(error)


def main() -> None:
    st.set_page_config(
        page_title="Signal | Misinformation Research",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(f"<style>{read_asset('streamlit.css')}</style>")

    state = st.session_state
    for key in ("analysis", "last_request_id", "input_error"):
        state.setdefault(key, None)

    result = dashboard_component(
        read_asset("index.html"),
        read_asset("styles.css"),
        read_asset("app.js"),
    )(
        key="signal",
        data={
            "analysis": state.analysis,
            "last_request_id": state.last_request_id,
            "error": state.input_error,
            "max_characters": MAX_CHARACTERS,
        },
        on_request_change=lambda: None,
    )
    request = result.request
    if not request or request.get("id") == state.last_request_id:
        return
    process_request(request)
    st.rerun()


if __name__ == "__main__":
    main()
