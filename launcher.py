"""
Deployment entry point.
=======================

Runs the simulator. That is the whole job.

    python -m streamlit run launcher.py

This file used to offer a sidebar control that switched the deployment between
the enhanced build and the original one. It no longer does: the deployed app
serves the enhanced version and nothing else, with no way to switch away from
it, so that anyone opening the published link sees one application rather than
a choice they have no basis to make.

`app_co2_units_fixed.py` is still in the repository and still runs on its own
if you want to compare the two locally:

    python -m streamlit run app_co2_units_fixed.py

Nothing here modifies the application: it is executed exactly as it would be if
run directly. The only interference is that `set_page_config` is neutralised
for the inner script, because Streamlit permits that call once per session and
this launcher has already made it.
"""
import os
import runpy

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
APP = "streamlit_app.py"

st.set_page_config(
    page_title="Remineralization Simulator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

target = os.path.join(HERE, APP)
if not os.path.exists(target):
    st.error(f"**{APP} is missing.** It must sit next to this launcher.")
    st.stop()

# Streamlit allows set_page_config once per session and the launcher has used
# it, so it is disabled for the duration of the inner script.
_real_set_page_config = st.set_page_config
st.set_page_config = lambda *a, **k: None
try:
    runpy.run_path(target, run_name="__main__")
finally:
    st.set_page_config = _real_set_page_config
