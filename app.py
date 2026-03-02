"""
Streamlit dashboard for the London Underground Delay Prediction system.

TfL-branded interface with live predictions, training diagnostics,
per-line comparisons, and data collection status.

This file is the entry point for `streamlit run app.py`.
All logic lives in the app/ package — this shim just wires
st.set_page_config (which must be the first Streamlit call) to main().
"""

import streamlit as st
import sys
from pathlib import Path

# Make sure the project root is on sys.path so app/ can import config, utils, etc.
sys.path.insert(0, str(Path(__file__).parent))

# PAGE CONFIG must be the very first Streamlit call
st.set_page_config(
    layout="wide",
    page_title="TfL Delay Predictor | ML Dashboard",
    page_icon="🚇",
    initial_sidebar_state="expanded",
)

from app import main

if __name__ == "__main__":
    main()
