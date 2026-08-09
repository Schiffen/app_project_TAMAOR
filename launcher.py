"""
Version switcher.
=================

An optional entry point that can serve EITHER version of the simulator from a
single deployment, so the enhanced build and the original can be compared side
by side — or the original can be made the live one again — without editing any
code or redeploying.

    python -m streamlit run launcher.py

Pick the version from the control at the top of the sidebar. The choice is
written to the URL (`?app=original`), so a particular version can be linked to
or bookmarked directly.

Nothing here modifies either application: each one is executed exactly as it
would be if run on its own. The only interference is that `set_page_config` is
neutralised for the inner app, because Streamlit permits that call once per
session and this launcher has already made it.

You do NOT have to use this file. Running either application directly still
works, and on Streamlit Community Cloud the "Main file path" setting can point
at any of the three entry points:

    streamlit_app.py         enhanced version only
    app_co2_units_fixed.py   original version only
    launcher.py              both, switchable at runtime
"""
import os
import runpy

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

VERSIONS = {
    "enhanced": {
        "label": "Enhanced simulator",
        "file": "streamlit_app.py",
        "note": "Activity-corrected chemistry, self-consistent feed water, "
                "interactive charts.",
    },
    "original": {
        "label": "Original (unmodified)",
        "file": "app_co2_units_fixed.py",
        "note": "The project exactly as originally written. Nothing altered.",
    },
}
DEFAULT = "enhanced"

st.set_page_config(
    page_title="Remineralization Simulator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The URL is the source of truth, so a version can be linked to directly.
_q = st.query_params.get("app")
if _q not in VERSIONS:
    _q = DEFAULT
if st.session_state.get("_version") != _q:
    st.session_state["_version"] = _q

keys = list(VERSIONS)
with st.sidebar:
    chosen = st.radio(
        "Version",
        keys,
        index=keys.index(st.session_state["_version"]),
        format_func=lambda k: VERSIONS[k]["label"],
        key="_version_radio",
        help="Switch between the enhanced build and the untouched original. "
             "Both read the same inputs; only the model and interface differ.",
    )
    st.caption(VERSIONS[chosen]["note"])
    st.divider()

if chosen != st.session_state["_version"]:
    st.session_state["_version"] = chosen
    st.query_params["app"] = chosen
    st.rerun()

target = os.path.join(HERE, VERSIONS[chosen]["file"])
if not os.path.exists(target):
    st.error(f"**{VERSIONS[chosen]['file']} is missing.** Both application files "
             f"must sit next to this launcher.")
    st.stop()

# Streamlit allows set_page_config once per session and the launcher has used
# it, so it is disabled for the duration of the inner script.
_real_set_page_config = st.set_page_config
st.set_page_config = lambda *a, **k: None
try:
    runpy.run_path(target, run_name="__main__")
finally:
    st.set_page_config = _real_set_page_config
