# NOTE:
# This file originally contained a Google Colab export of the training notebook.
# Streamlit cannot execute notebook-only commands like:
#   - `pip install ...`
#   - `!unzip ...` / `!ls -R`
#   - `from google.colab import files`
#
# For deployment, the actual Streamlit UI lives in `app.py`.
# This stub keeps the file importable so Streamlit (or any runner that points
# to this path) won’t fail with SyntaxError.

import streamlit as st

st.set_page_config(page_title="Student Performance Prediction", page_icon="📚", layout="wide")

st.warning(
    "This deployment uses `app.py` as the Streamlit entry point. "
    "Redirecting…"
)

# Try to embed/redirect by linking to the correct app URL.
# (In most Streamlit hosting setups, the entry script is `app.py` already.)
try:
    st.page_link("../app.py", label="Open app.py", icon="📚")
except Exception:
    pass

st.stop()

