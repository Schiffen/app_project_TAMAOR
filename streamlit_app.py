"""
Remineralization & Water Transport Simulator
============================================

Simulates the post-treatment train that turns desalinated permeate into
distributed drinking water, and tracks calcium-carbonate stability along it:

    Desalinated water -> Remineralization -> Closed reservoir -> Supply pipe -> Consumer

Model contents
--------------
* Carbonate equilibrium (H2CO3* / HCO3- / CO3^2-) with Van't Hoff temperature
  correction of K1, K2, Kw, Ksp and KH.
* IONIC STRENGTH: activity coefficients from the Davies equation, applied as
  conditional (concentration-based) equilibrium constants and iterated to
  self-consistency with the speciation. pH is reported on the activity scale
  and SI is built from ion activities. At I = 0 the model reduces exactly to
  the infinite-dilution case.
* Gas-liquid MASS TRANSFER of CO2 between the stored water and a closed
  headspace (kLa driving force towards Henry's-law equilibrium).
* Precipitation KINETICS of CaCO3 in the supply pipe, driven by the local
  saturation index, with simultaneous heat loss to the surroundings.
* Saturation Index (SI) and Calcium Carbonate Precipitation Potential (CCPP).

The initial water is specified by any TWO of pH, alkalinity and total inorganic
carbon; the third is calculated, because the three are not independent.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq, minimize

# =============================================================================
# 1. PAGE + DESIGN TOKENS
# =============================================================================
st.set_page_config(
    page_title="Remineralization Simulator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Three-state axis for the carbonate condition of the water — the one thing
# this whole tool computes. Blue = the water dissolves carbonate, green = it is
# inside the band the drinking-water standard asks for, red = it deposits.
C_AGGR = "#0891B2"          # undersaturated / aggressive
C_BALANCED = "#15803D"      # inside the target band
C_SCALE = "#C2410C"         # oversaturated / scaling
C_NEUTRAL = "#94A3B8"       # not available

# The three above are RESERVED: they encode carbonate condition and nothing
# else. Categorical series therefore draw from the remaining hue space, so a
# line on a chart can never be mistaken for a state.
SERIES = ["#4F46E5", "#A21CAF", "#334E5C"]     # indigo, fuchsia, graphite
PRIMARY = "#334E5C"                            # buttons / neutral marks

INK = "#0B1F27"
INK_MID = "#4A6470"
INK_MUTED = "#5F7C87"
LINE = "#DCE7EC"
CARD = "#FFFFFF"

# Illustration palette. Kept clear of the reserved three so a picture can never
# be read as a state reading. Water is ONE blue everywhere in the train; what
# changes from stage to stage is what is dissolved in it, drawn as calcite
# grains in a limestone cream. That is the actual subject of the simulation.
W_LIGHT, W_MID, W_DEEP = "#8ED3F2", "#2E8FCB", "#12496E"
MIN_FILL, MIN_EDGE = "#F2DCB0", "#B8863C"      # calcite / limestone
STEEL, STEEL_DARK = "#8FA6B2", "#33505E"       # equipment
GAS_FILL = "#EAF2F6"                           # headspace

st.markdown(
    f"""
<style>
/* IBM Plex was drawn for engineering and technical products; the mono cut
   carries the measured values, which keeps digits aligned and column-scannable. */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
[data-testid="stAppViewContainer"] {{ background: #F4F8FA; }}
.block-container {{ max-width: 1480px; padding-top: 2.1rem; padding-bottom: 3rem; }}
/* The sidebar width is deliberately NOT pinned here. Streamlit writes its own
   width as an inline style (300px, and whatever the reader drags it to) and
   slides the panel shut by translating it by exactly that inline figure. Any
   min/max-width from this stylesheet is therefore only honoured on the way in:
   a 306px panel translated by 300px leaves a 6px strip of sidebar frozen on
   screen, which is what made "closed" look broken rather than closed. */
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2 {{ font-size: .74rem !important; text-transform: uppercase;
    letter-spacing: .1em; color: {INK_MUTED}; font-weight: 600; margin: .1rem 0 .3rem 0; }}
[data-testid="stSidebar"] label {{ font-size: .8rem !important; }}
[data-testid="stSidebar"] hr {{ margin: 1.05rem 0; }}

/* Streamlit's own vendor chrome has no place on a page being read as a product.
   It is hidden by its ACTIONS, though, never by the whole toolbar: the toolbar
   is also where Streamlit puts stExpandSidebarButton, the single control that
   reopens a collapsed sidebar. Hiding the toolbar outright left that button in
   the DOM at zero size, and because the collapsed state is remembered in
   localStorage it survives every reload — so one accidental collapse shut the
   sidebar permanently, with nothing on the page able to bring it back. */
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stExpandSidebarButton"] {{ opacity: .75; }}
[data-testid="stExpandSidebarButton"]:hover {{ opacity: 1; }}

/* Streamlit wraps headings in its own container whose rule outranks a bare
   h1/h3 selector, so the scale has to be addressed through that wrapper. */
[data-testid="stHeadingWithActionElements"] h1 {{
    font-size: 2.15rem !important; font-weight: 700; line-height: 1.1;
    letter-spacing: -0.028em; color: {INK}; }}
[data-testid="stHeadingWithActionElements"] h3 {{
    font-size: .8rem !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: .11em; color: {INK_MUTED} !important; margin: 2.2rem 0 .5rem 0; }}
[data-testid="stTab"], [data-baseweb="tab"] {{
    font-size: .875rem !important; font-weight: 600 !important; }}

.lede {{ color: {INK_MID}; font-size: .98rem; line-height: 1.62; max-width: 60ch;
    margin: .7rem 0 0 0; }}

/* ---- wordmark ---------------------------------------------------------- */
.brand {{ display:flex; align-items:center; gap:1.05rem; }}
.brand svg {{ flex:none; }}
h1.wm {{ font-size: 2.15rem; font-weight: 700; line-height: 1.16; letter-spacing: -.03em;
    color: {INK}; margin: 0; padding: .06em 0 0 0; }}

/* ---- process train ---------------------------------------------------- */
/* Five tiles with an arrow gutter between each pair. The gutters are real
   grid tracks rather than absolutely-positioned decoration, so the tiles stay
   equal width and the arrows can never overlap a label. */
/* The train is laid out with st.columns rather than one grid of raw HTML,
   because each tile has to be a real Streamlit button. The tile track and the
   arrow gutter alternate, so the weights below mirror the old 1fr / 30px grid.
   Streamlit's own column gap is tightened here; the spacing that reads as the
   flow gutter is the arrow column itself, not the gap. */
.st-key-train [data-testid="stHorizontalBlock"] {{ gap:.34rem; align-items:stretch; }}
.st-key-train [data-testid="stColumn"] {{ position:relative; }}
/* Streamlit pulls markdown blocks up with a -16px bottom margin, which left the
   column measuring 16px shorter than the card inside it — and therefore left the
   bottom strip of every tile outside the overlay button and dead to the click. */
.st-key-train [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
.stage {{ text-decoration:none; display:flex; flex-direction:column; align-items:center;
    padding:.85rem .5rem .7rem .5rem; border:1px solid {LINE}; border-radius:14px;
    background:{CARD}; position:relative;
    transition: transform .22s cubic-bezier(.2,.8,.3,1),
                box-shadow .22s cubic-bezier(.2,.8,.3,1), border-color .2s ease; }}
.stage svg {{ flex:none; }}
.stage.on {{ border-color: transparent;
    box-shadow: inset 0 0 0 2px var(--cond), 0 16px 30px -20px rgba(11,31,39,.5); }}

/* Each tile is a drawn card with a Streamlit button laid invisibly over the
   whole of it. The click therefore reruns the script over the open websocket
   instead of navigating the browser to a new URL, which is what previously
   tore down and rebuilt the entire page — sidebar and session state with it.
   The button keeps its accessible name, so the tile is still a real, focusable,
   keyboard-operable control; only its painted surface is suppressed. */
/* width/height are forced to auto so that inset:0 is what actually sizes the
   overlay: Streamlit gives every element container an explicit width, and an
   explicit width beats left:0/right:0, which otherwise leaves the tile's outer
   rim unclickable. */
div[class*="st-key-stagebtn-"] {{ position:absolute !important; inset:0 !important;
    width:auto !important; height:auto !important; max-width:none !important;
    margin:0 !important; z-index:3; }}
div[class*="st-key-stagebtn-"] .stButton,
div[class*="st-key-stagebtn-"] [data-testid="stButton"] {{ width:100%; height:100%; }}
div[class*="st-key-stagebtn-"] button {{ width:100% !important; height:100% !important;
    min-height:0 !important; opacity:0; padding:0; border:0; background:transparent;
    cursor:pointer; }}

/* Hover and focus have to be driven from the column, not from .stage: the
   pointer rests on the overlay button, which is a sibling of the card rather
   than a child, so .stage:hover would never match. */
.st-key-train [data-testid="stColumn"]:has(div[class*="st-key-stagebtn-"]):hover .stage {{
    transform: translateY(-3px); border-color:#C4D8E0;
    box-shadow: 0 16px 30px -22px rgba(11,31,39,.55); }}
.st-key-train [data-testid="stColumn"]:has(div[class*="st-key-stagebtn-"] button:focus-visible) .stage {{
    outline:2px solid var(--cond); outline-offset:3px; }}

/* the arrow gutter: a hairline rail with a travelling dash, so the train
   reads as a flow rather than five unrelated cards */
.flow {{ display:flex; align-items:center; justify-content:center; }}
.flow svg {{ overflow:visible; }}
.flow .rail {{ stroke:{LINE}; stroke-width:2; }}
.flow .run {{ stroke:{W_MID}; stroke-width:2; stroke-linecap:round;
    stroke-dasharray:5 13; animation: drift 1.5s linear infinite; }}
.flow .head {{ fill:{W_MID}; }}
@keyframes drift {{ to {{ stroke-dashoffset:-18; }} }}
@media (prefers-reduced-motion: reduce) {{
    .flow .run {{ animation:none; stroke-dasharray:none; opacity:.55; }}
}}

.stage, .stage * {{ text-decoration: none !important; }}
.stage-name {{ text-align:center; font-size:.83rem; font-weight:600;
    color:{INK} !important; margin-top:.45rem; letter-spacing:-.01em; }}
.stage-si {{ text-align:center; font-size:.9rem; color:{INK_MUTED} !important;
    margin-top:.28rem; font-family:'IBM Plex Mono',ui-monospace,monospace;
    font-variant-numeric: tabular-nums; }}
.stage-si b {{ color: var(--cond) !important; font-weight:600; }}
.stage-ccpp {{ text-align:center; font-size:.72rem; margin-top:.12rem;
    color:{INK_MUTED} !important; font-family:'IBM Plex Mono',ui-monospace,monospace;
    font-variant-numeric: tabular-nums; }}

/* Below the width where five tiles can hold a legible label, the train wraps
   and the arrow gutters are dropped rather than squeezed. Streamlit's column row
   is already a wrapping flex container, so it is enough to give the tiles a
   160px basis — five of them left in one row would be 97px each, narrow enough
   that the drawn units spill out past the card edge — and to remove the arrow
   columns entirely, since an arrow between two wrapped tiles points nowhere. */
@media (max-width: 1120px) {{
    .flow {{ display:none; }}
    .st-key-train [data-testid="stColumn"]:not(:has(div[class*="st-key-stagebtn-"])) {{
        display:none; }}
    .st-key-train [data-testid="stColumn"]:has(div[class*="st-key-stagebtn-"]) {{
        flex: 1 1 160px !important; }}
    .st-key-train [data-testid="stHorizontalBlock"] {{ gap:.5rem; }}
}}

/* ---- hero reading ------------------------------------------------------ */
/* The saturation index is the one number this whole application computes, so
   it gets the focal point rather than being the tail of a sentence. */
.cond {{ display:grid; grid-template-columns: 196px 1fr; align-items:center;
    gap:1.1rem; padding:.9rem 1.2rem; border:1px solid {LINE};
    border-radius:13px; background:{CARD}; margin:.1rem 0 1rem 0; }}
.cond-k {{ font-size:.68rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
    color:{INK_MUTED}; }}
.cond-v {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:3.05rem;
    font-weight:600; line-height:1.02; letter-spacing:-.04em; color:var(--cond);
    font-variant-numeric: tabular-nums; }}
/* SI names the direction, CCPP names the amount, so the two are read together
   rather than on separate surfaces. Neither decides the verdict any more. */
.cond-ccpp {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:.8rem;
    color:{INK_MID}; margin-top:.3rem; font-variant-numeric: tabular-nums; }}
.cond-ccpp b {{ color:{INK}; font-weight:600; }}

/* ---- drinking-water criteria ------------------------------------------- */
/* The verdict is a summary of five separate tests, so the tests are shown.
   A reader who is told the water is unfit should not have to guess which
   criterion failed, or by how much. */
.crit {{ display:flex; flex-wrap:wrap; gap:.36rem; margin-top:.6rem; }}
.crit .c {{ display:inline-flex; align-items:baseline; gap:.34rem; font-size:.75rem;
    padding:.2rem .52rem .24rem .52rem; border-radius:8px; border:1px solid; }}
.crit .c i {{ font-style:normal; font-weight:600; }}
.crit .c b {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-weight:600;
    font-variant-numeric: tabular-nums; }}
.crit .c s {{ text-decoration:none; opacity:.72; font-size:.68rem; }}
.crit .ok   {{ background:#F1F9F4; border-color:#C9E6D3; color:#14532D; }}
.crit .low  {{ background:#EDF7FB; border-color:#C2E2ED; color:#0D4C60; }}
.crit .high {{ background:#FDF2ED; border-color:#F2D2C1; color:#7C2D12; }}
.crit .na   {{ background:#F5F7F8; border-color:{LINE}; color:{INK_MUTED}; }}
.cond-unit {{ font-size:1.28rem; font-weight:600; color:{INK}; letter-spacing:-.02em; }}
.cond-txt {{ font-size:.97rem; color:{INK}; font-weight:600; margin-top:.12rem; }}
.cond-sub {{ font-size:.86rem; color:{INK_MID}; font-weight:400; margin-top:.1rem;
    max-width: 46ch; }}

/* ---- instrument readout ----------------------------------------------- */
/* Fixed four-track module: 8 values fill exactly two flush rows, so the
   vertical rules line up and the card closes cleanly. */
.readout {{ display:grid; grid-template-columns:repeat(4, 1fr);
    border:1px solid {LINE}; border-radius:13px; background:{CARD}; overflow:hidden;
    animation: rise .32s cubic-bezier(.2,.8,.3,1); }}
@media (max-width: 820px)  {{ .readout {{ grid-template-columns:repeat(2, 1fr); }} }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(5px); }}
                   to {{ opacity:1; transform:none; }} }}
.cell {{ padding:.62rem .8rem .66rem .8rem; border-right:1px solid {LINE};
    border-bottom:1px solid {LINE}; }}
.cell.hl {{ background:#FBFDFE; }}
/* Deliberately NOT uppercased: these labels carry element symbols and Greek
   letters (Ca²⁺, Mg²⁺, pH) that uppercasing would render incorrectly. */
.cell .k {{ font-size:.735rem; letter-spacing:.028em; color:{INK_MUTED};
    font-weight:600; white-space:nowrap; }}
.cell .v {{ font-family:'IBM Plex Mono',ui-monospace,monospace; font-size:1.1rem;
    font-weight:600; color:{INK}; margin-top:.18rem; font-variant-numeric: tabular-nums;
    letter-spacing:-.02em; white-space:nowrap; }}
.cell .u {{ font-family:'IBM Plex Sans',sans-serif; font-size:.72rem; color:{INK_MUTED};
    font-weight:500; margin-left:.2rem; }}

.note {{ font-size:.84rem; color:{INK_MUTED}; line-height:1.55; }}
hr {{ border-color:{LINE}; }}

/* charts join the same container language as every other surface */
[data-testid="stPlotlyChart"] {{ border:1px solid {LINE}; border-radius:13px;
    background:{CARD}; overflow:hidden; }}
.figmark {{ max-width: 300px; opacity:.66; margin:.2rem 0 .1rem 0; }}
.charttitle {{ font-size:.72rem; font-weight:600; letter-spacing:.085em; text-transform:uppercase;
    color:{INK_MUTED}; margin:.7rem 0 .35rem 0; }}

/* ---- model assumptions ------------------------------------------------- */
/* Rendered as a definition list rather than markdown bullets. Streamlit sets
   `code` spans in a green monospace at a smaller size than their surrounding
   text, which broke every constant onto its own visual register and wrapped
   the exponents mid-value. Symbol and value are now two aligned columns, and
   powers of ten are typeset as powers of ten. */
.assump {{ display:grid; grid-template-columns:auto 1fr; gap:.32rem .7rem;
    margin:.1rem 0 .2rem 0; }}
.assump dt {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:.78rem;
    font-weight:600; color:{INK}; white-space:nowrap; }}
.assump dd {{ margin:0; font-size:.78rem; color:{INK_MID}; line-height:1.45;
    font-variant-numeric: tabular-nums; }}
.assump dd .n {{ font-family:'IBM Plex Mono', ui-monospace, monospace;
    font-weight:600; color:{INK}; }}
.assump-note {{ font-size:.76rem; color:{INK_MUTED}; line-height:1.5;
    margin:.7rem 0 0 0; padding-top:.6rem; border-top:1px solid {LINE}; }}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 2. PHYSICAL CONSTANTS AND MODEL PARAMETERS
# =============================================================================
R = 8.314462618           # J mol-1 K-1
R_GAS = 8.205736e-5       # m3 atm mol-1 K-1
RHO_W = 1000.0            # kg m-3
CP_W = 4180.0             # J kg-1 K-1

MW = {
    "CaOH2": 74.09268, "CO2": 44.0095, "CaCO3": 100.0869, "MgCl2": 95.211,
    "Ca": 40.078, "Mg": 24.305, "NaCl": 58.44277, "Cl": 35.453,
}

# Thermodynamic (infinite-dilution) constants at 25 C
K1_25, K2_25 = 10 ** -6.35, 10 ** -10.33
KW_25, KSP_25 = 1e-14, 10 ** -8.48
KH_25 = 0.0339            # mol L-1 atm-1

# Van't Hoff reaction enthalpies [J/mol]
DH_K1, DH_K2, DH_KW, DH_KSP, DH_KH = 7.7e3, 14.9e3, 55.8e3, -9.4e3, -19.4e3

KLA = 1.0e-4              # s-1, volumetric gas-liquid mass transfer coefficient
KS_SCALE = 1.0e-8         # mol L-1 s-1 per unit positive SI
U_EXPOSED, U_INSULATED = 8.0, 1.0     # W m-2 K-1
PCO2_GAS_0 = 420e-6       # atm, initial headspace CO2 partial pressure

# Background salinity of the permeate, mg/L as NaCl. Fixed rather than exposed:
# it sets the chloride level (sodium follows from electroneutrality) and feeds
# the ionic-strength correction, but it is a property of the RO plant, not an
# operating decision the person using this model gets to make.
NACL_BACKGROUND = 60.0

# ---- what counts as water fit to drink -------------------------------------
# The condition of the water is judged on whether a person can drink it, not on
# what it will do to the inside of the pipe. Those are different questions and
# they do not have the same answer: water can be perfectly protective of a
# cement lining and still be too soft, too flat and too low in calcium to be
# supplied, which is exactly the failure mode of unremineralized desalinated
# permeate.
#
# Each criterion below is a published drinking-water figure, not a modelling
# choice. Bands are (minimum, maximum); None means that side is unbounded.
#
#   pH 7.0-8.5        WHO / US EPA secondary standard. Below 7 the water is
#                     corrosive and flat-tasting, above 8.5 it turns soapy.
#   Calcium >= 32     Israeli Ministry of Health minimum for desalinated
#                     drinking water (equivalently 80 mg/L as CaCO3).
#   Magnesium >= 10   WHO health-based recommendation. Desalinated permeate is
#                     essentially magnesium-free, and it has to be put back.
#   Alkalinity >= 80  Israeli MoH minimum. This is the buffering that stops the
#                     pH of the delivered water from moving in the network.
#   Hardness 60-180   WHO palatability classification: below 60 the water is
#                     "soft" and tastes empty; 60-120 is moderately hard and
#                     120-180 is hard, both perfectly drinkable; only above
#                     180 does it start furring kettles and appliances.
#
#                     The maximum CANNOT be set at 120 even though that is the
#                     more familiar number, because hardness is not independent
#                     of the two criteria above it: TH = 2.497*Ca + 4.118*Mg,
#                     so the calcium and magnesium minimums by themselves force
#                     TH >= 2.497*32 + 4.118*10 = 121 mg/L. A 120 ceiling would
#                     make the five criteria mutually unsatisfiable and no dose
#                     set could ever pass them.
#
# The optimisation target for each criterion sits inside its band, and the
# scale is the amount of that quantity that counts as one unit of "off". Total
# hardness carries no target: it is a consequence of calcium and magnesium
# rather than something dosed on its own, so it is constrained but not aimed at.
DRINK_CRITERIA = [
    # label,            reads,                    min,   max,   target, scale, unit,            dp
    ("pH",              lambda s: s["pH"],        7.0,   8.5,   7.75,   0.75,  "",               2),
    ("Calcium",         lambda s: s["Ca_mg"],     32.0,  None,  40.0,   8.0,   "mg/L",           1),
    ("Magnesium",       lambda s: s["Mg_mg"],     10.0,  None,  14.0,   4.0,   "mg/L",           1),
    ("Alkalinity",      lambda s: s["Alk_mg"],    80.0,  None,  100.0,  20.0,  "mg/L as CaCO₃",  1),
    ("Total hardness",  lambda s: s["TH"],        60.0,  180.0, None,   40.0,  "mg/L as CaCO₃",  1),
]

# Still reported, because it is what the model computes about the pipe — but no
# longer what decides the verdict.
CCPP_MIN_MG, CCPP_MAX_MG = 3.0, 10.0      # mg/L as CaCO3

DOSE_MAX = {"CaOH2": 200.0, "CO2": 200.0, "CaCO3": 300.0, "MgCl2": 200.0}
CHEM_LABEL = {"CaOH2": "Ca(OH)₂", "CO2": "CO₂", "CaCO3": "CaCO₃", "MgCl2": "MgCl₂"}
LABEL_CHEM = {v: k for k, v in CHEM_LABEL.items()}

# =============================================================================
# 3. CHEMISTRY CORE  (activity-corrected)
# =============================================================================
def vanthoff(Kref, dH, T_C):
    return Kref * math.exp((-dH / R) * (1 / (T_C + 273.15) - 1 / 298.15))


def thermo_constants(T_C):
    """Equilibrium constants at infinite dilution, temperature corrected."""
    return {
        "K1": vanthoff(K1_25, DH_K1, T_C), "K2": vanthoff(K2_25, DH_K2, T_C),
        "Kw": vanthoff(KW_25, DH_KW, T_C), "Ksp": vanthoff(KSP_25, DH_KSP, T_C),
        "KH": vanthoff(KH_25, DH_KH, T_C),
    }


def dh_A(T_C):
    """Debye-Huckel A parameter; ~0.51 (mol/L)^-0.5 at 25 C.

    Dielectric constant of water from Malmberg & Maryott (1956).
    """
    eps = 87.740 - 0.40008 * T_C + 9.398e-4 * T_C ** 2 - 1.410e-6 * T_C ** 3
    return 1.82e6 / (eps * (T_C + 273.15)) ** 1.5


def davies_gamma(z, I, A):
    """Davies activity coefficient for charge z at ionic strength I (to ~0.5 M)."""
    if I <= 0:
        return 1.0
    s = math.sqrt(I)
    return 10 ** (-A * z * z * (s / (1 + s) - 0.3 * I))


def conditional_constants(T_C, I):
    """Thermodynamic K -> concentration-based K' at ionic strength I.

        K1  = g1^2 [H][HCO3]/[H2CO3*]   ->  K1'  = K1 / g1^2
        K2  = g2   [H][CO3]/[HCO3]      ->  K2'  = K2 / g2
        Kw  = g1^2 [H][OH]              ->  Kw'  = Kw / g1^2
        Ksp = g2^2 [Ca][CO3]            ->  Ksp' = Ksp / g2^2

    H2CO3* is uncharged, so its activity coefficient is taken as 1.
    """
    K = thermo_constants(T_C)
    A = dh_A(T_C)
    g1, g2 = davies_gamma(1, I, A), davies_gamma(2, I, A)
    return {
        "K1": K["K1"] / g1 ** 2, "K2": K["K2"] / g2, "Kw": K["Kw"] / g1 ** 2,
        "Ksp": K["Ksp"] / g2 ** 2, "KH": K["KH"],
        "g1": g1, "g2": g2, "Ksp_T": K["Ksp"],
    }


def speciate(H, CT, Kc):
    """Carbonate distribution from free [H+]."""
    den = H * H + Kc["K1"] * H + Kc["K1"] * Kc["K2"]
    a0 = H * H / den
    a1 = Kc["K1"] * H / den
    a2 = Kc["K1"] * Kc["K2"] / den
    return a0, a1, a2, a0 * CT, a1 * CT, a2 * CT


def alk_from_H(H, CT, Kc):
    _, _, _, _, hco3, co3 = speciate(H, CT, Kc)
    return hco3 + 2 * co3 + Kc["Kw"] / H - H


def solve_H(Alk, CT, Kc):
    """Free [H+] from the alkalinity balance.

    alk_from_H is strictly decreasing in [H+], so the root is unique and one
    bracketed solve over the whole pH range is enough.
    """
    lo, hi = 1e-14, 1e-1
    if (alk_from_H(lo, CT, Kc) - Alk) * (alk_from_H(hi, CT, Kc) - Alk) > 0:
        return float("nan")
    return brentq(lambda h: alk_from_H(h, CT, Kc) - Alk, lo, hi, xtol=1e-22, rtol=1e-12)


def _spectators(Ca, Mg, Alk, Cl):
    """Electroneutrality fixes the spectator ions.

    2[Ca]+2[Mg]+[Na]+[H] = [Cl]+[HCO3]+2[CO3]+[OH],  and since
    Alk = [HCO3]+2[CO3]+[OH]-[H],   [Na] = Alk - 2[Ca] - 2[Mg] + [Cl].
    """
    Na = Alk - 2 * Ca - 2 * Mg + Cl
    return (Na, Cl) if Na >= 0 else (0.0, Cl - Na)


def ionic_strength(Ca, Mg, Na, Cl, H, OH, HCO3, CO3):
    return 0.5 * (4 * Ca + 4 * Mg + Na + Cl + H + OH + HCO3 + 4 * CO3)


def hardness(Ca_mg, Mg_mg):
    return 2.497 * Ca_mg + 4.118 * Mg_mg


def build_state(name, Ca, Mg, CT, Alk, Cl, T_C, itmax=30, tol=1e-10):
    """Self-consistent speciation, ionic strength and saturation index."""
    Na, Cl_eff = _spectators(Ca, Mg, Alk, Cl)

    I = 0.0
    for _ in range(itmax):
        Kc = conditional_constants(T_C, I)
        H = solve_H(Alk, CT, Kc)
        if math.isnan(H):
            break
        _, _, _, _, hco3, co3 = speciate(H, CT, Kc)
        I_new = ionic_strength(Ca, Mg, Na, Cl_eff, H, Kc["Kw"] / H, hco3, co3)
        if abs(I_new - I) < tol:
            I = I_new
            break
        I = I_new

    Kc = conditional_constants(T_C, I)
    H = solve_H(Alk, CT, Kc)

    nan = float("nan")
    if math.isnan(H):
        pH = SI = nan
        a0 = a1 = a2 = co2 = hco3 = co3 = nan
    else:
        a0, a1, a2, co2, hco3, co3 = speciate(H, CT, Kc)
        pH = -math.log10(Kc["g1"] * H)                    # activity scale
        SI = (math.log10((Kc["g2"] * Ca) * (Kc["g2"] * co3) / Kc["Ksp_T"])
              if Ca > 0 and co3 > 0 else nan)

    Ca_mg, Mg_mg = Ca * MW["Ca"] * 1000, Mg * MW["Mg"] * 1000
    return {
        "name": name, "pH": pH, "SI": SI, "I": I,
        "g1": Kc["g1"], "g2": Kc["g2"],
        "Alk": Alk, "Alk_mg": Alk * 50000.0,
        "Ca": Ca, "Ca_mg": Ca_mg, "Mg": Mg, "Mg_mg": Mg_mg,
        "TH": hardness(Ca_mg, Mg_mg), "CT": CT, "Cl": Cl, "T": T_C,
        "CO2": co2, "HCO3": hco3, "CO3": co3, "a0": a0, "a1": a1, "a2": a2,
    }


def CT_from_pH_alk(pH, Alk, Ca, Mg, Cl, T_C, itmax=30):
    """Total inorganic carbon implied by a measured pH and alkalinity."""
    Na, Cl_eff = _spectators(Ca, Mg, Alk, Cl)
    I, CT = 0.0, 0.0
    for _ in range(itmax):
        Kc = conditional_constants(T_C, I)
        H = 10 ** (-pH) / Kc["g1"]
        den = H * H + Kc["K1"] * H + Kc["K1"] * Kc["K2"]
        a1 = Kc["K1"] * H / den
        a2 = Kc["K1"] * Kc["K2"] / den
        carb = Alk - Kc["Kw"] / H + H
        if carb <= 0:
            return float("nan")
        CT_new = carb / (a1 + 2 * a2)
        _, _, _, _, hco3, co3 = speciate(H, CT_new, Kc)
        I_new = ionic_strength(Ca, Mg, Na, Cl_eff, H, Kc["Kw"] / H, hco3, co3)
        if abs(CT_new - CT) < 1e-14 and abs(I_new - I) < 1e-12:
            return CT_new
        CT, I = CT_new, I_new
    return CT


def alk_from_pH_CT(pH, CT, Ca, Mg, Cl, T_C, itmax=30):
    """Alkalinity implied by a measured pH and total inorganic carbon."""
    I, Alk = 0.0, 0.0
    for _ in range(itmax):
        Kc = conditional_constants(T_C, I)
        H = 10 ** (-pH) / Kc["g1"]
        _, _, _, _, hco3, co3 = speciate(H, CT, Kc)
        Alk_new = hco3 + 2 * co3 + Kc["Kw"] / H - H
        Na, Cl_eff = _spectators(Ca, Mg, Alk_new, Cl)
        I_new = ionic_strength(Ca, Mg, Na, Cl_eff, H, Kc["Kw"] / H, hco3, co3)
        if abs(Alk_new - Alk) < 1e-16 and abs(I_new - I) < 1e-12:
            return Alk_new
        Alk, I = Alk_new, I_new
    return Alk


def solve_ccpp(s):
    """CaCO3 that must precipitate (+) or dissolve (-) to reach SI = 0.

    Removing y mol/L of CaCO3 removes y from Ca and CT and 2y from alkalinity.
    """
    if np.isnan(s["SI"]):
        return float("nan")
    if abs(s["SI"]) < 1e-8:
        return 0.0

    Ca0, CT0, Alk0, Cl, T = s["Ca"], s["CT"], s["Alk"], s["Cl"], s["T"]
    Mg0 = s["Mg"]          # magnesium is a spectator for CaCO3, but it carries
                           # charge, so it must stay in the ionic-strength sum

    def g(y):
        Ca, CT, Alk = Ca0 - y, CT0 - y, Alk0 - 2 * y
        if Ca <= 0 or CT <= 0:
            return float("nan")
        return build_state("t", Ca, Mg0, CT, Alk, Cl, T)["SI"]

    if s["SI"] > 0:
        hi = min(Ca0, CT0) * 0.999
        if Alk0 > 0:
            hi = min(hi, Alk0 / 2 * 0.999)
        if hi <= 0:
            return float("nan")
        ys = np.linspace(0, hi, 60)
    else:
        ys = np.linspace(-0.02, 0, 60)

    prev_y = prev_f = None
    for y in ys:
        fy = g(y)
        if np.isnan(fy):
            continue
        if abs(fy) < 1e-9:
            return float(y)
        if prev_f is not None and prev_f * fy < 0:
            return float(brentq(g, prev_y, y, xtol=1e-12))
        prev_y, prev_f = y, fy
    return float("nan")


def with_ccpp(s):
    out = dict(s)
    out["CCPP"] = solve_ccpp(s)
    return out


def ccpp_mg(s):
    """CCPP in mg/L as CaCO3 — the unit the drinking-water standard is written in."""
    if s is None:
        return float("nan")
    v = s.get("CCPP")
    if v is None or np.isnan(v):
        return float("nan")
    return v * MW["CaCO3"] * 1000.0


def drink_report(s):
    """Every drinking-water criterion, its value, and which way it fails.

    Rows are (label, value, unit, dp, side); side is 'ok', 'low', 'high', 'na'.
    """
    rows = []
    for label, read, lo, hi, _t, _sc, unit, dp in DRINK_CRITERIA:
        v = read(s) if s is not None else float("nan")
        if v is None or np.isnan(v):
            side = "na"
        elif lo is not None and v < lo:
            side = "low"
        elif hi is not None and v > hi:
            side = "high"
        else:
            side = "ok"
        rows.append((label, v, unit, dp, side))
    return rows


def classify(s):
    """Is this water fit to supply — and if not, which way is it wrong?

    'good'  every criterion met
    'hard'  something is ABOVE its maximum: over-mineralized, hard, soapy
    'soft'  otherwise something is BELOW its minimum: under-mineralized and
            aggressive, which is the untreated-permeate failure

    A water failing on both sides at once is reported as over-mineralized: that
    is the fault you dose *down* to fix, and it is the one a consumer notices.
    """
    if s is None:
        return None
    sides = [r[4] for r in drink_report(s)]
    if "na" in sides:
        return None
    if all(x == "ok" for x in sides):
        return "good"
    return "hard" if "high" in sides else "soft"


# =============================================================================
# 4. PROCESS UNITS
# =============================================================================
def dose_to_mol(dose_mg_L, mw):
    return dose_mg_L / (mw * 1000.0)


def remineralize(initial, doses, compute_ccpp=True):
    """Lime, CO2, calcite and MgCl2 addition.

    Ca(OH)2 -> Ca2+ + 2 OH-     : +1 Ca, +2 alk
    CaCO3   -> Ca2+ + CO3^2-    : +1 Ca, +1 CT, +2 alk
    CO2 + H2O -> H2CO3*         : +1 CT,  no alk
    MgCl2   -> Mg2+ + 2 Cl-     : +1 Mg, +2 Cl, no alk
    """
    n_lime = dose_to_mol(doses["CaOH2"], MW["CaOH2"])
    n_co2 = dose_to_mol(doses["CO2"], MW["CO2"])
    n_cal = dose_to_mol(doses["CaCO3"], MW["CaCO3"])
    n_mg = dose_to_mol(doses["MgCl2"], MW["MgCl2"])

    s = build_state(
        "Remineralization outlet",
        initial["Ca"] + n_lime + n_cal,
        initial["Mg"] + n_mg,
        initial["CT"] + n_co2 + n_cal,
        initial["Alk"] + 2 * n_lime + 2 * n_cal,
        initial["Cl"] + 2 * n_mg,
        initial["T"],
    )
    return with_ccpp(s) if compute_ccpp else s


def reservoir(inlet, t_h, Vw_m3, Vhs_m3, T_C, compute_ccpp=True, nsteps=60,
              vented=False):
    """Storage reservoir: CO2 exchange across the gas-liquid interface.

    Driving force is always kLa*(CO2*_aq - KH*pCO2). The two modes differ only
    in what happens to the gas phase:

    CLOSED  - a sealed headspace of finite volume. Every mole leaving the water
              enters that headspace, so pCO2 moves and the transfer self-limits
              once the two phases equilibrate. With a large water volume and a
              small headspace this happens almost immediately.

    VENTED  - open to atmosphere. The air is effectively an infinite reservoir,
              so pCO2 stays at ambient and transfer continues for as long as the
              water is held. This is how most service reservoirs actually run,
              and it is where residence time genuinely changes the water.
    """
    Ca, Mg, CT, Alk, Cl = inlet["Ca"], inlet["Mg"], inlet["CT"], inlet["Alk"], inlet["Cl"]
    T_K = T_C + 273.15
    Vw_L = Vw_m3 * 1000.0
    n_gas = PCO2_GAS_0 * Vhs_m3 / (R_GAS * T_K)
    KH = thermo_constants(T_C)["KH"]

    t_s = t_h * 3600.0
    if t_s > 0:
        # Keep kLa*dt well below 1 so the explicit step stays accurate. This
        # only engages beyond ~30 h of residence time; shorter runs are
        # unaffected and reproduce the original integration exactly.
        nsteps = max(4, int(nsteps))
        nsteps = max(nsteps, min(600, int(t_s / 1800) + 1))
        dt = t_s / nsteps
        Cstar_open = KH * PCO2_GAS_0
        for _ in range(nsteps):
            s = build_state("t", Ca, Mg, CT, Alk, Cl, T_C)
            if np.isnan(s["CO2"]):
                break
            if vented:
                dC = KLA * (s["CO2"] - Cstar_open) * dt
                dC = min(dC, CT * 0.20) if dC > 0 else max(dC, -CT * 0.20)
                CT = max(CT - dC, 1e-15)
            else:
                pco2 = n_gas * R_GAS * T_K / max(Vhs_m3, 1e-12)
                dC = KLA * (s["CO2"] - KH * pco2) * dt
                dC = min(dC, CT * 0.20) if dC > 0 else max(dC, -n_gas / max(Vw_L, 1e-30))
                CT = max(CT - dC, 1e-15)
                n_gas = max(n_gas + dC * Vw_L, 0.0)

    label = "Vented reservoir outlet" if vented else "Closed reservoir outlet"
    out = build_state(label, Ca, Mg, CT, Alk, Cl, T_C)
    if compute_ccpp:
        out = with_ccpp(out)
    pco2_out = (PCO2_GAS_0 if vented
                else n_gas * R_GAS * T_K / max(Vhs_m3, 1e-12))
    return out, {"pCO2": pco2_out, "residence_h": t_h, "vented": vented}


def pipe(inlet, L, D, Q_m3h, Tenv, insulated, nseg=80, compute_ccpp=True,
         want_profile=True):
    """Supply pipe: heat loss to ambient plus SI-driven CaCO3 deposition."""
    Q = Q_m3h / 3600.0
    A = math.pi * D ** 2 / 4.0
    v = Q / A
    ttot = L / v

    U = U_INSULATED if insulated else U_EXPOSED
    kT = 4 * U / (RHO_W * CP_W * D)

    Ca, Mg, CT, Alk, Cl, T = (inlet["Ca"], inlet["Mg"], inlet["CT"],
                              inlet["Alk"], inlet["Cl"], inlet["T"])
    dt, dx = ttot / nseg, L / nseg
    cum = 0.0
    rows = []

    for i in range(nseg + 1):
        s = build_state("pipe", Ca, Mg, CT, Alk, Cl, T)
        if want_profile:
            rows.append({"x": i * dx, "pH": s["pH"], "SI": s["SI"], "T": T,
                         "cum_precip": cum})
        if i == nseg:
            break

        T = Tenv + (T - Tenv) * math.exp(-kT * dt)
        s2 = build_state("pipe", Ca, Mg, CT, Alk, Cl, T)
        rate = KS_SCALE * max(s2["SI"], 0.0) if not np.isnan(s2["SI"]) else 0.0
        dy = rate * dt
        dy = min(dy, Ca * 0.999, CT * 0.999)
        if Alk > 0:
            dy = min(dy, Alk / 2 * 0.999)
        dy = max(dy, 0.0)

        Ca -= dy
        CT -= dy
        Alk -= 2 * dy
        cum += dy

    out = build_state("Pipe outlet / Consumer", Ca, Mg, CT, Alk, Cl, T)
    if compute_ccpp:
        out = with_ccpp(out)
    profile = pd.DataFrame(rows) if want_profile else None
    return out, profile, {"A": A, "v": v, "t_s": ttot, "U": U, "precip": cum}


def make_initial(mode, pH0, alk0_mg, CT0, Ca0_mg, Mg0_mg, nacl_mg, T0):
    """Build the feed water from any TWO of pH / alkalinity / C_T.

    The carbonate system links the three, so only two can be specified
    independently; the third is calculated and reported back.
    """
    Ca = Ca0_mg / (MW["Ca"] * 1000.0)
    Mg = Mg0_mg / (MW["Mg"] * 1000.0)
    Cl = nacl_mg / (MW["NaCl"] * 1000.0)          # background Cl-, Na+ closes charge

    if mode == "pH + alkalinity":
        Alk = alk0_mg / 50000.0
        CT = CT_from_pH_alk(pH0, Alk, Ca, Mg, Cl, T0)
        derived = ("Cₜ", "mmol/L", CT * 1000 if not np.isnan(CT) else float("nan"))
        if np.isnan(CT) or CT <= 0:
            # Carbonate alkalinity = Alk - [OH-] + [H+]. If water's own hydroxide
            # already exceeds the specified alkalinity, no carbonate can balance it.
            Kc = conditional_constants(T0, 0.0)
            H = 10 ** (-pH0) / Kc["g1"]
            oh_mg = (Kc["Kw"] / H - H) * 50000.0
            return None, derived, (
                f"**This feed water cannot exist.** At pH {pH0:.2f}, hydroxide alone "
                f"already contributes **{oh_mg:.0f} mg/L as CaCO₃** of alkalinity — more "
                f"than the **{alk0_mg:.0f} mg/L** you specified. Carbonate would have to "
                f"be negative. Lower the pH, or raise the alkalinity above {oh_mg:.0f} mg/L."
            )
    elif mode == "pH + Cₜ":
        CT = CT0
        Alk = alk_from_pH_CT(pH0, CT, Ca, Mg, Cl, T0)
        derived = ("Alkalinity", "mg/L as CaCO₃", Alk * 50000.0)
    else:                                          # alkalinity + C_T
        Alk, CT = alk0_mg / 50000.0, CT0
        derived = ("pH", "", float("nan"))

    if np.isnan(CT) or CT <= 0:
        return None, derived, (
            "**This feed water cannot exist.** The pH, alkalinity and Cₜ you specified "
            "have no consistent solution. Try a different specification mode."
        )

    s = with_ccpp(build_state("Desalinated water", Ca, Mg, CT, Alk, Cl, T0))
    if np.isnan(s["pH"]):
        return None, derived, (
            f"**No pH satisfies this water.** An alkalinity of {alk0_mg:.0f} mg/L as CaCO₃ "
            f"cannot be carried by only {CT*1000:.3f} mmol/L of inorganic carbon. "
            f"Raise Cₜ or lower the alkalinity."
        )
    if mode == "alkalinity + Cₜ":
        derived = ("pH", "", s["pH"])
    return s, derived, None


# =============================================================================
# 5. CACHED SIMULATION
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=512)
def simulate(mode, pH0, alk0, CT0, Ca0, Mg0, nacl, T0, doses_t,
             t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated,
             res_steps=60, nseg=80, ccpp=True, profile=True):
    """Whole train for one set of inputs. Cached, so repeated reruns are free."""
    doses = dict(zip(("CaOH2", "CO2", "CaCO3", "MgCl2"), doses_t))
    init, derived, problem = make_initial(mode, pH0, alk0, CT0, Ca0, Mg0, nacl, T0)
    if init is None:
        return {"problem": problem}
    rem = remineralize(init, doses, ccpp)
    res, res_x = reservoir(rem, t_res, Vw, Vhs, Tres, ccpp, res_steps, vented)
    out, prof, pipe_x = pipe(res, L, D, Q, Tenv, insulated, nseg, ccpp, profile)
    return {"initial": init, "remin": rem, "res": res, "pipe": out,
            "profile": prof, "res_x": res_x, "pipe_x": pipe_x, "derived": derived,
            "problem": None}


def consumer_only(mode, pH0, alk0, CT0, Ca0, Mg0, nacl, T0, doses,
                  t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated, fast=True):
    """Consumer state alone, at reduced resolution — for scans and optimisation.

    The train runs without CCPP and it is solved once, on the consumer state
    alone. CCPP is the quantity the target band is defined on, so a search that
    skipped it would be optimising the wrong number — but the intermediate
    stages are not being scored here, and solving it for them as well would
    triple the cost of every evaluation for nothing.
    """
    r = simulate(mode, pH0, alk0, CT0, Ca0, Mg0, nacl, T0,
                 (doses["CaOH2"], doses["CO2"], doses["CaCO3"], doses["MgCl2"]),
                 t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated,
                 res_steps=14 if fast else 60, nseg=10 if fast else 80,
                 ccpp=False, profile=False)
    if r is None or r.get("problem"):
        return None
    return with_ccpp(r["pipe"])


# =============================================================================
# 6. AUTOMATIC STABILISATION
# =============================================================================
class StablePointFound(Exception):
    def __init__(self, x, state):
        self.x = np.asarray(x, dtype=float)
        self.state = state


def is_stable(s):
    """Every drinking-water criterion met."""
    return classify(s) == "good"


def is_comfortably_stable(s, margin=0.2):
    """Met, and not sitting on a boundary.

    The search stops the moment it finds acceptable water, which left it
    returning points balanced on the edge of a band — hardness at 179.7 against
    a ceiling of 180. Water like that is technically a pass and practically
    useless, because the next nudge of any input tips it back out. The early
    exit therefore asks for a margin of a fifth of each criterion's scale.
    """
    if s is None:
        return False
    for _lbl, read, lo, hi, _t, scale, _u, _dp in DRINK_CRITERIA:
        v = read(s)
        if v is None or np.isnan(v):
            return False
        if lo is not None and v < lo + margin * scale:
            return False
        if hi is not None and v > hi - margin * scale:
            return False
    return True


def quality(s):
    """Lower is better: how far outside its band each criterion sits.

    Distances are divided by that criterion's own scale so a pH unit and a
    mg/L of calcium can be added together, and the sum is zero anywhere inside
    every band — no point in the acceptable region is preferred to another.
    """
    if s is None:
        return np.inf
    q = 0.0
    for _lbl, read, lo, hi, _t, scale, _u, _dp in DRINK_CRITERIA:
        v = read(s)
        if v is None or np.isnan(v):
            return np.inf
        if lo is not None and v < lo:
            q += (lo - v) / scale
        elif hi is not None and v > hi:
            q += (v - hi) / scale
    return float(q)


DOSE_VARS = ("CaOH2", "CO2", "CaCO3", "MgCl2")


def auto_stabilize(base, current, fast_first=True):
    """Search the four doses for water that meets every drinking criterion.

    MgCl2 is searched alongside the other three because magnesium is one of the
    criteria and nothing else in the train supplies it — desalinated permeate
    carries essentially none, so holding MgCl2 fixed would leave the magnesium
    minimum permanently unreachable.

    Powell with a hard evaluation cap, an early exit as soon as every criterion
    is met, then a full-resolution validation of whatever it found.
    """
    def evaluate(x, fast):
        d = {k: float(v) for k, v in zip(DOSE_VARS, x)}
        return consumer_only(**base, doses=d, fast=fast), d

    span = np.array([DOSE_MAX[k] for k in DOSE_VARS], dtype=float)

    def objective(x, fast):
        s, _ = evaluate(x, fast)
        if s is None or np.isnan(s["pH"]):
            return 1e6
        if is_comfortably_stable(s):
            raise StablePointFound(x, s)
        # A purely one-sided penalty is flat everywhere inside the bands and
        # gives Powell nothing to follow, so a weak pull toward the middle of
        # each band rides on top of it. The hard term dominates whenever any
        # criterion is actually violated.
        score = 0.0
        for _lbl, read, lo, hi, target, scale, _u, _dp in DRINK_CRITERIA:
            v = read(s)
            if v is None or np.isnan(v):
                return 1e6
            if lo is not None and v < lo:
                score += 4.0 * ((lo - v) / scale) ** 2
            elif hi is not None and v > hi:
                score += 4.0 * ((v - hi) / scale) ** 2
            if target is not None:
                score += 0.05 * ((v - target) / scale) ** 2
        ref = np.array([current[k] for k in DOSE_VARS], dtype=float)
        score += 0.015 * np.sum(((np.asarray(x) - ref) / span) ** 2)
        return float(score)

    bounds = [(0.0, DOSE_MAX[k]) for k in DOSE_VARS]
    x0 = np.array([current[k] for k in DOSE_VARS], dtype=float)

    now, _ = evaluate(x0, False)
    if is_stable(now):
        return {"doses": dict(current), "state": now, "stable": True,
                "message": "The current operating point already meets the target."}

    def powell(start, fast, maxfev):
        try:
            r = minimize(objective, np.asarray(start, float), args=(fast,),
                         method="Powell", bounds=bounds,
                         options={"maxfev": maxfev, "maxiter": 50,
                                  "xtol": 0.5, "ftol": 1e-4, "disp": False})
            return np.asarray(r.x, float)
        except StablePointFound as hit:
            return hit.x

    x_fast = powell(x0, fast_first, 160)
    cand = [(current, now)]
    for x in (x_fast, powell(x_fast, False, 80)):
        d = {k: float(np.clip(v, 0, DOSE_MAX[k])) for k, v in zip(DOSE_VARS, x)}
        cand.append((d, consumer_only(**base, doses=d, fast=False)))

    best_d, best_s = min(cand, key=lambda p: quality(p[1]))
    return {"doses": dict(best_d), "state": best_s, "stable": is_stable(best_s),
            "message": ("Every drinking-water criterion is met." if is_stable(best_s)
                        else "No dose set meeting every criterion was found within "
                             "the evaluation limit; the closest one is shown.")}


# =============================================================================
# 7. FIGURES
# =============================================================================
COND_COLOR = {"soft": C_AGGR, "good": C_BALANCED, "hard": C_SCALE}
COND_TEXT = {
    "soft": ("Not fit to drink — under-mineralized",
             "Too soft and too lightly buffered to be supplied. Water like this "
             "tastes empty and dissolves the cement lining it travels through."),
    "good": ("Good for drinking",
             "Meets every criterion for supplied drinking water: calcium, "
             "magnesium, alkalinity, hardness and pH are all in range."),
    "hard": ("Not fit to drink — over-mineralized",
             "Carrying more mineral than drinking water should. Water like this "
             "tastes soapy and furs kettles, pipes and appliances."),
}


def cond_color(s):
    """Discrete state colour: under-mineralized / good / over-mineralized."""
    return COND_COLOR.get(classify(s), C_NEUTRAL)


MINUS = "\u2212"          # U+2212, not a hyphen: correct in numeric settings


def fmt(v, nd=2):
    """Format a reading, using a true minus sign rather than a hyphen."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}".replace("-", MINUS)


def fmt_signed(v, nd=2):
    """Signed reading (SI), with a real plus/minus glyph."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}".replace("-", MINUS)


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v, nd=1):
    """A power of ten typeset as a power of ten, not as `1.0e-04`.

    Python's e-notation is a serialisation format, not a way of setting a
    number for someone to read: it pads the exponent, spells the multiplication
    as a letter, and in a proportional face the mantissa and exponent end up on
    two different visual registers.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    exp = int(math.floor(math.log10(abs(v)))) if v else 0
    mant = v / (10 ** exp)
    return f"{mant:.{nd}f} × 10{str(exp).translate(SUP)}"


def style(fig, xtitle, ytitle, height=380, legend=False):
    fig.update_layout(
        template="simple_white", height=height,
        margin=dict(l=58, r=16, t=16, b=46),
        font=dict(family="IBM Plex Sans, sans-serif", size=13, color=INK_MID),
        hoverlabel=dict(font_family="IBM Plex Mono, monospace", font_size=12,
                        bgcolor="white", bordercolor=LINE),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        plot_bgcolor=CARD, paper_bgcolor=CARD,
    )
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=12, color=INK_MUTED)),
                     showgrid=True, gridcolor="#EFF4F6", gridwidth=1,
                     linecolor=LINE, ticks="outside", tickcolor=LINE,
                     tickfont=dict(size=12, color=INK_MUTED), zeroline=False)
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=12, color=INK_MUTED)),
                     showgrid=True, gridcolor="#EFF4F6", gridwidth=1,
                     linecolor=LINE, ticks="outside", tickcolor=LINE,
                     tickfont=dict(size=12, color=INK_MUTED), zeroline=False)
    return fig


def add_si_zero(fig):
    """The line the whole simulation is aimed at."""
    fig.add_hline(y=0, line=dict(color=C_NEUTRAL, width=1.5, dash="dash"),
                  annotation_text="calcite equilibrium",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=INK_MUTED))


def line_fig(x, y, xtitle, ytitle, color=PRIMARY, si=False, height=380):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=color, width=2.5, shape="spline"),
        hovertemplate="%{y:.3f}<extra></extra>", name=ytitle))
    if si:
        add_si_zero(fig)
    return style(fig, xtitle, ytitle, height)


def bjerrum_figure(T_C, I, pH_now, a_now):
    """Carbonate distribution across pH, with the current water marked."""
    Kc = conditional_constants(T_C, I)
    pHs = np.linspace(3, 12, 260)
    A = np.array([speciate(10 ** (-p) / Kc["g1"], 1.0, Kc)[:3] for p in pHs])

    names = ["CO₂* (H₂CO₃*)", "HCO₃⁻", "CO₃²⁻"]
    fig = go.Figure()
    for i, (nm, c) in enumerate(zip(names, SERIES[:3])):
        fig.add_trace(go.Scatter(
            x=pHs, y=A[:, i], mode="lines", name=nm,
            line=dict(color=c, width=2.5),
            hovertemplate=f"{nm}: %{{y:.3f}}<extra></extra>"))

    if not np.isnan(pH_now):
        fig.add_vline(x=pH_now, line=dict(color=INK_MUTED, width=1.5, dash="dot"))
        fig.add_annotation(x=pH_now, y=1.06, text=f"this water · pH {pH_now:.2f}",
                           showarrow=False, font=dict(size=11, color=INK),
                           xanchor="center")
        for i, c in enumerate(SERIES[:3]):
            if not np.isnan(a_now[i]):
                fig.add_trace(go.Scatter(
                    x=[pH_now], y=[a_now[i]], mode="markers", showlegend=False,
                    marker=dict(color=c, size=10, line=dict(color="white", width=2)),
                    hovertemplate=f"{names[i]}: %{{y:.3f}}<extra></extra>"))

    fig.update_yaxes(range=[-0.03, 1.13])
    return style(fig, "pH", "Fraction of total inorganic carbon", 400, legend=True)


def stage_figure(names, values, ytitle, si=False, marker_colors=None):
    colors = marker_colors or [SERIES[0]] * len(values)
    fig = go.Figure(go.Scatter(
        x=names, y=values, mode="lines+markers",
        line=dict(color=C_NEUTRAL, width=2),
        marker=dict(size=15, color=colors, line=dict(color="white", width=2.5)),
        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    if si:
        add_si_zero(fig)
    return style(fig, "", ytitle, 380)


# =============================================================================
# 8. PROCESS-UNIT ILLUSTRATIONS
# =============================================================================
def svg_logo(w=54):
    """A calcite rhomb held inside a drop of water — the subject in one mark.

    Neither half works alone: a drop is any water utility, a crystal is any
    mineralogy department. Together they are this specific problem, which is
    putting the mineral back into water that had it stripped out.

    The rhomb is drawn on the real crystallography rather than as a generic
    cube. Calcite is trigonal R3̄c; its {101̄4} cleavage rhomb has face angles
    of 78°05′ / 101°55′. The vertices below are the orthographic projection of
    that solid down a mirror-symmetric azimuth, so the silhouette is a true
    calcite rhomb and not a squashed hexagon. (The common error is drawing the
    faces at 74°55′ — that is the dihedral angle between two faces in three
    dimensions, not the angle you see.)
    """
    return f"""<svg width="{w}" height="{w * 1.2:.0f}" viewBox="0 0 100 120" fill="none">
      <defs>
        <linearGradient id="lgDrop" x1="0" y1="0" x2=".35" y2="1">
          <stop offset="0" stop-color="{W_LIGHT}"/>
          <stop offset=".52" stop-color="{W_MID}"/>
          <stop offset="1" stop-color="{W_DEEP}"/>
        </linearGradient>
      </defs>
      <path d="M50 4 C62 26 92 52 92 74 A42 42 0 0 1 8 74 C8 52 38 26 50 4 Z"
            fill="url(#lgDrop)"/>
      <path d="M27 46 C21 57 19 65 19 72" stroke="#FFFFFF" stroke-opacity=".5"
            stroke-width="4.4" stroke-linecap="round"/>
      <polygon points="50,64.93 31,68.82 31,89.17 50,85.27" fill="#FBF0DA"/>
      <polygon points="50,64.93 69,68.82 69,89.17 50,85.27" fill="#E2C48A"/>
      <polygon points="50,93.06 69,89.17 50,85.27 31,89.17" fill="{MIN_EDGE}"/>
      <path d="M50 85.27 L69 89.17 M50 85.27 L31 89.17 M50 85.27 L50 64.93"
            stroke="#8A6428" stroke-width="1.5" stroke-opacity=".55"/>
      <polygon points="50,93.06 69,89.17 69,68.82 50,64.93 31,68.82 31,89.17"
               fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linejoin="round"/>
    </svg>"""


# ---------------------------------------------------------------------------
# Process-unit illustrations.
#
# The earlier set was ISO 10628 line art: correct, but monochrome and almost
# unreadable at tile size, where every unit resolved to the same grey outline.
# These are drawn to be told apart at a glance instead, on one rule that keeps
# them honest as a set:
#
#     WATER IS ONE BLUE EVERYWHERE. What changes from stage to stage is what is
#     DISSOLVED IN IT, drawn as calcite grains in a limestone cream.
#
# So the train reads left to right as a story rather than five unrelated
# pictures: the RO vessel splits salts out and hands on water with nothing in
# it, the dosing station puts grains back, the reservoir holds them, the pipe
# lays some down on its own wall, and a glass arrives carrying the rest. The
# equipment stays flat line art in steel so the colour is never decoration —
# it is always either water or mineral.
#
# The illustration palette is kept clear of the reserved blue/green/red, so a
# picture can never be misread as a saturation reading.
# ---------------------------------------------------------------------------
ICON = ('<svg width="118" height="82" viewBox="0 0 132 92" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round">')
W_EQ, W_LN = 2.6, 2.2


def _grains(pts, r=2.7):
    """Calcite in suspension."""
    return "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{MIN_FILL}" '
                   f'stroke="{MIN_EDGE}" stroke-width="1"/>' for x, y in pts)


def _tri(x, y, d="r", fill=None, s=5.0):
    """Flow arrowhead, 1.6:1 length to base."""
    fill = fill or STEEL_DARK
    p = (f"{x},{y-s} {x+s*1.6:.1f},{y} {x},{y+s}" if d == "r"
         else f"{x-s},{y} {x},{y+s*1.6:.1f} {x+s},{y}")
    return f'<polygon points="{p}" fill="{fill}"/>'


def svg_flow():
    """The connector between two stages: a rail with a dash travelling down it.

    The train is a flow, so it is drawn as one. The dash is the only motion on
    the page and it is switched off under prefers-reduced-motion.
    """
    return ('<svg width="30" height="22" viewBox="0 0 30 22" fill="none">'
            '<path class="rail" d="M1 11 H21"/>'
            '<path class="run" d="M1 11 H21"/>'
            '<polygon class="head" points="19,5.5 29,11 19,16.5"/>'
            '</svg>')


def svg_desal():
    """Reverse-osmosis vessel: salty feed in, salts rejected, bare water out.

    The membrane is the dashed barrier down the middle, and the whole point of
    the drawing is that grains appear on its left and never on its right. That
    is what makes the water that leaves aggressive, and it is the reason every
    other unit in this train exists.
    """
    return f"""{ICON}
    <defs>
      <linearGradient id="gDeF" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{W_MID}"/><stop offset="1" stop-color="{W_DEEP}"/>
      </linearGradient>
      <linearGradient id="gDeP" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#E8F6FD"/><stop offset="1" stop-color="{W_LIGHT}"/>
      </linearGradient>
    </defs>
    <path d="M2 46 H18" stroke="{STEEL_DARK}" stroke-width="{W_LN}"/>{_tri(12, 46)}
    <path d="M66 26 H36 A20 20 0 0 0 36 66 H66 Z" fill="url(#gDeF)"/>
    <path d="M66 26 H96 A20 20 0 0 1 96 66 H66 Z" fill="url(#gDeP)"/>
    {_grains([(30, 38), (42, 55), (35, 47), (54, 36), (50, 58), (59, 47)])}
    <path d="M66 24 V68" stroke="#FFFFFF" stroke-width="3.4" stroke-dasharray="5 4"/>
    <path d="M36 26 H96 A20 20 0 0 1 96 66 H36 A20 20 0 0 1 36 26 Z"
          stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    <path d="M46 66 V78" stroke="{STEEL_DARK}" stroke-width="{W_LN}"/>{_tri(46, 76, "d")}
    <path d="M116 46 H128" stroke="{STEEL_DARK}" stroke-width="{W_LN}"/>{_tri(122, 46)}
    </svg>"""


def svg_remin():
    """Dosing station over the process line: mineral goes back into the water.

    A day hopper of calcite and lime feeding an injection point, with grains
    falling into the pipe and carried on downstream of it. The pipe is clear on
    the left of the quill and carries mineral on the right — the addition is
    the whole unit, so the drawing puts it on the centre line.
    """
    return f"""{ICON}
    <defs>
      <linearGradient id="gReH" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#FBF0DA"/><stop offset="1" stop-color="{MIN_FILL}"/>
      </linearGradient>
      <linearGradient id="gReW" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{W_LIGHT}"/><stop offset="1" stop-color="{W_MID}"/>
      </linearGradient>
    </defs>
    <polygon points="42,6 90,6 79,32 53,32" fill="url(#gReH)"
             stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    {_grains([(55, 15), (66, 12), (77, 16), (61, 24), (72, 25)], 2.5)}
    <path d="M66 32 V44" stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    {_grains([(66, 49), (63, 57), (69, 64)], 2.4)}
    <rect x="2" y="55" width="128" height="22" fill="url(#gReW)"/>
    <path d="M2 55 H130 M2 77 H130" stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    {_grains([(80, 62), (93, 70), (105, 61), (116, 69)], 2.6)}
    {_tri(26, 66, "r", "#FFFFFF", 5.4)}{_tri(46, 66, "r", "#FFFFFF", 5.4)}
    </svg>"""


def svg_res():
    """Storage tank: water under a gas headspace it exchanges CO₂ with.

    The headspace is drawn as a real volume rather than the single line a P&ID
    would use, because in this model it is a volume — a finite one, whose pCO₂
    moves as CO₂ leaves the water, which is what makes the transfer stop.
    """
    return f"""{ICON}
    <defs>
      <linearGradient id="gRsW" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{W_MID}"/><stop offset="1" stop-color="{W_DEEP}"/>
      </linearGradient>
    </defs>
    <path d="M4 30 H24" stroke="{STEEL_DARK}" stroke-width="{W_LN}"/>{_tri(18, 30)}
    <path d="M24 24 H108 V80 H24 Z" fill="{GAS_FILL}"/>
    <path d="M24 46 Q45 39 66 46 T108 46 V80 H24 Z" fill="url(#gRsW)"/>
    <circle cx="43" cy="35" r="3.4" fill="#FFFFFF" stroke="{STEEL}" stroke-width="1.2"/>
    <circle cx="58" cy="31" r="2.4" fill="#FFFFFF" stroke="{STEEL}" stroke-width="1.2"/>
    <circle cx="72" cy="36" r="2.9" fill="#FFFFFF" stroke="{STEEL}" stroke-width="1.2"/>
    <circle cx="88" cy="31" r="2.1" fill="#FFFFFF" stroke="{STEEL}" stroke-width="1.2"/>
    {_grains([(40, 60), (62, 68), (86, 57), (74, 73), (50, 72), (96, 69)])}
    <polygon points="20,24 66,8 112,24" fill="{STEEL}" fill-opacity=".35"
             stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    <path d="M24 24 V80 H108 V24" stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    <path d="M108 70 H128" stroke="{STEEL_DARK}" stroke-width="{W_LN}"/>{_tri(122, 70)}
    </svg>"""


def svg_pipe():
    """Supply run in longitudinal section, with scale growing along it.

    The cream layer thickens from left to right because that is what the model
    computes: deposition is cumulative, so the far end of the run carries more
    of it than the near end. The channel narrowing is the consequence a network
    operator actually cares about.
    """
    return f"""{ICON}
    <defs>
      <linearGradient id="gPpW" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{W_LIGHT}"/><stop offset=".5" stop-color="{W_MID}"/>
        <stop offset="1" stop-color="{W_DEEP}"/>
      </linearGradient>
    </defs>
    <rect x="2" y="28" width="128" height="36" fill="url(#gPpW)"/>
    <polygon points="34,28 130,28 130,35.5 34,29" fill="{MIN_FILL}"/>
    <polygon points="34,64 130,64 130,56.5 34,63" fill="{MIN_FILL}"/>
    <path d="M34 29 L130 35.5 M34 63 L130 56.5" stroke="{MIN_EDGE}" stroke-width="1.3"/>
    <path d="M2 28 H130 M2 64 H130" stroke="{STEEL_DARK}" stroke-width="3"/>
    {_tri(24, 46, "r", "#FFFFFF", 5.4)}{_tri(54, 46, "r", "#FFFFFF", 5.4)}
    {_tri(84, 46, "r", "#FFFFFF", 5.4)}
    </svg>"""


def svg_consumer():
    """The tap the whole train is aimed at.

    Water arrives carrying the mineral that was put back into it upstream. The
    glass is the only unit in the train nobody operates, which is the point:
    everything before it exists to make this one right.
    """
    return f"""{ICON}
    <defs>
      <linearGradient id="gCnW" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{W_LIGHT}"/><stop offset="1" stop-color="{W_MID}"/>
      </linearGradient>
    </defs>
    <path d="M28 6 V22 H69 V31" stroke="{STEEL_DARK}" stroke-width="5"/>
    <path d="M19 11 H37" stroke="{STEEL_DARK}" stroke-width="4"/>
    <path d="M69 38 c3.4 4.4 5.4 6.8 5.4 8.8 a5.4 5.4 0 0 1 -10.8 0 c0 -2 2 -4.4 5.4 -8.8 z"
          fill="{W_MID}"/>
    <polygon points="49.5,63 88.5,63 85,86 53,86" fill="url(#gCnW)"/>
    {_grains([(62, 72), (76, 79), (69, 68), (57, 81)], 2.5)}
    <polygon points="46,54 92,54 86,88 52,88" stroke="{STEEL_DARK}" stroke-width="{W_EQ}"/>
    <path d="M49.5 63 H88.5" stroke="#FFFFFF" stroke-width="2" stroke-opacity=".75"/>
    </svg>"""


# =============================================================================
# 9. SIDEBAR
# =============================================================================
# Doses live in session_state so the optimiser can write them. They are seeded
# here rather than passed as widget defaults, because a widget that declares a
# default AND is assigned through session_state triggers a Streamlit warning.
DOSE_KEYS = {"dose_lime": 20.0, "dose_co2": 15.0, "dose_calcite": 30.0, "dose_mgcl2": 5.0}
for _k, _v in DOSE_KEYS.items():
    st.session_state.setdefault(_k, _v)

# Optimised doses must land in session_state BEFORE the widgets are created.
if "pending_doses" in st.session_state:
    p = st.session_state.pop("pending_doses")
    st.session_state["dose_lime"] = float(p["CaOH2"])
    st.session_state["dose_co2"] = float(p["CO2"])
    st.session_state["dose_calcite"] = float(p["CaCO3"])
    st.session_state["dose_mgcl2"] = float(p["MgCl2"])

with st.sidebar:
    st.header("Desalinated water")
    spec_mode = st.selectbox(
        "Specified by", ["pH + alkalinity", "pH + Cₜ", "alkalinity + Cₜ"],
        help="pH, alkalinity and Cₜ are linked by the carbonate equilibrium, "
             "so only two can be set independently. The third is calculated.",
    )
    show_pH = spec_mode in ("pH + alkalinity", "pH + Cₜ")
    show_alk = spec_mode in ("pH + alkalinity", "alkalinity + Cₜ")
    show_CT = spec_mode in ("pH + Cₜ", "alkalinity + Cₜ")

    pH0 = st.number_input("pH", 3.0, 11.0, 6.50, 0.05) if show_pH else 6.50
    alk0 = st.number_input("Alkalinity [mg/L as CaCO₃]", 0.0, 500.0, 20.0, 1.0) if show_alk else 20.0
    CT0 = (st.number_input("Cₜ [mol/L]", 1e-6, 0.05, 5e-4, 5e-5, format="%.6f")
           if show_CT else 5e-4)

    Ca0 = st.number_input("Ca²⁺ [mg/L]", 0.0, 500.0, 5.0, 1.0)
    Mg0 = st.number_input("Mg²⁺ [mg/L]", 0.0, 500.0, 1.0, 0.5)
    T0 = st.number_input("Temperature [°C]", 1.0, 60.0, 25.0, 1.0)

    st.divider()
    st.header("Remineralization")
    d_lime = st.slider("Ca(OH)₂ [mg/L]", 0.0, DOSE_MAX["CaOH2"], step=1.0, key="dose_lime")
    d_co2 = st.slider("CO₂ [mg/L]", 0.0, DOSE_MAX["CO2"], step=1.0, key="dose_co2")
    d_calc = st.slider("CaCO₃ [mg/L]", 0.0, DOSE_MAX["CaCO3"], step=1.0, key="dose_calcite")
    d_mg = st.slider("MgCl₂ [mg/L]", 0.0, DOSE_MAX["MgCl2"], step=1.0, key="dose_mgcl2")

    st.divider()
    st.header("Closed reservoir")
    t_res = st.number_input("Residence time [h]", 0.0, 72.0, 2.0, 0.25)
    Vw = st.number_input("Water volume [m³]", 0.01, 1e7, 1000.0, 10.0)
    Vhs = st.number_input("Headspace volume [m³]", 0.01, 1e7, 100.0, 10.0,
                          help="The sealed gas volume above the water. CO₂ leaving "
                               "the water raises its pCO₂, which is what brings the "
                               "transfer to a stop.")
    Tres = st.number_input("Reservoir temperature [°C]", 1.0, 60.0, 25.0, 1.0)

    st.divider()
    st.header("Supply pipe")
    L = st.number_input("Length [m]", 1.0, 1e7, 5000.0, 100.0)
    D = st.number_input("Diameter [m]", 0.01, 10.0, 0.50, 0.05)
    Q = st.number_input("Flow rate [m³/h]", 0.01, 1e7, 500.0, 10.0)
    Tenv = st.number_input("Ambient temperature [°C]", -10.0, 60.0, 30.0, 1.0)
    insulated = st.toggle("Insulated pipe", False)

    with st.expander("Model assumptions"):
        st.markdown(
            f'<dl class="assump">'
            f'<dt>k<sub>L</sub>a</dt><dd><span class="n">{sci(KLA)}</span> s⁻¹ '
            f'— gas-liquid transfer</dd>'
            f'<dt>k<sub>s</sub></dt><dd><span class="n">{sci(KS_SCALE)}</span> '
            f'mol L⁻¹ s⁻¹ per unit SI — deposition rate</dd>'
            f'<dt>U</dt><dd><span class="n">{U_EXPOSED:.0f}</span> exposed / '
            f'<span class="n">{U_INSULATED:.0f}</span> insulated W m⁻² K⁻¹</dd>'
            f'<dt>pCO₂</dt><dd><span class="n">{sci(PCO2_GAS_0)}</span> atm '
            f'— initial headspace</dd>'
            f'<dt>Salinity</dt><dd>permeate background held at '
            f'<span class="n">{NACL_BACKGROUND:.0f}</span> mg/L as NaCl</dd>'
            f'</dl>'
            f'<p class="assump-note"><b>Fit to drink</b> — every one of these '
            f'must hold at the consumer:</p>'
            f'<dl class="assump">'
            + "".join(
                f'<dt>{lbl}</dt><dd>'
                + (f'<span class="n">{lo:g}</span>–<span class="n">{hi:g}</span>'
                   if lo is not None and hi is not None else
                   f'≥ <span class="n">{lo:g}</span>' if lo is not None else
                   f'≤ <span class="n">{hi:g}</span>')
                + (f' {unit}' if unit else '') + '</dd>'
                for lbl, _r, lo, hi, _t, _sc, unit, _dp in DRINK_CRITERIA
            ) +
            f'</dl>'
            f'<p class="assump-note">K₁, K₂, K<sub>w</sub>, K<sub>sp</sub> and '
            f'K<sub>H</sub> are temperature-corrected with Van\'t Hoff. Activity '
            f'coefficients come from the Davies equation, applied as conditional '
            f'constants and iterated with the ionic strength; pH is reported on the '
            f'activity scale and SI is built from ion activities. The reservoir gas '
            f'phase is a finite sealed headspace.</p>',
            unsafe_allow_html=True,
        )

# =============================================================================
# 10. RUN
# =============================================================================
doses = {"CaOH2": d_lime, "CO2": d_co2, "CaCO3": d_calc, "MgCl2": d_mg}
base_inputs = dict(mode=spec_mode, pH0=pH0, alk0=alk0, CT0=CT0, Ca0=Ca0, Mg0=Mg0,
                   nacl=NACL_BACKGROUND, T0=T0, t_res=t_res, Vw=Vw, Vhs=Vhs, Tres=Tres,
                   vented=False, L=L, D=D, Q=Q, Tenv=Tenv, insulated=insulated)

sim = simulate(spec_mode, pH0, alk0, CT0, Ca0, Mg0, NACL_BACKGROUND, T0,
               (d_lime, d_co2, d_calc, d_mg),
               t_res, Vw, Vhs, Tres, False, L, D, Q, Tenv, insulated)

HEAD_LEDE = (
    '<p class="lede">Enter the initial water quality and operating conditions in the '
    'left sidebar, then adjust the remineralization doses, reservoir conditions and '
    'pipe parameters. Click any process-unit icon to inspect the calculated water '
    'quality at that stage, and use the analysis tabs below to explore how the '
    'selected inputs affect the process outputs.</p>'
)
HEAD_BRAND = (f'<div class="brand">{svg_logo(46)}'
              f'<div><h1 class="wm">Remineralization<br>&amp; water transport</h1></div></div>')

if sim is None or sim.get("problem"):
    st.markdown(HEAD_BRAND + HEAD_LEDE, unsafe_allow_html=True)
    st.error(sim["problem"] if sim else "This desalinated water cannot be solved.")
    st.stop()

initial, remin, res, pipeout = sim["initial"], sim["remin"], sim["res"], sim["pipe"]
profile = sim["profile"]
d_name, d_unit, d_val = sim["derived"]

st.markdown(HEAD_BRAND + HEAD_LEDE, unsafe_allow_html=True)

# Four stages, because the model computes four distinct states. The supply pipe
# is not a fifth state — it is the transport BETWEEN the reservoir and the tap,
# and its own metrics (velocity, travel time, deposition) appear on the outlet.
# Showing it as a separate tile reporting identical numbers to the Consumer, as
# an earlier layout did, invites the reader to click and find nothing changed.
STAGES = [
    ("Desalinated water", svg_desal(), "Initial", "initial", initial),
    ("Remineralization", svg_remin(), "Remineralization", "remineralization", remin),
    ("Closed reservoir", svg_res(), "Closed reservoir", "reservoir", res),
    ("Supply pipe", svg_pipe(), "Supply pipe", "pipe", pipeout),
    ("Consumer", svg_consumer(), "Consumer", "consumer", pipeout),
]
KEY_TO_LABEL = {k: lbl for lbl, _, k, _, _ in STAGES}
KEY_TO_LABEL["Supply pipe"] = "Consumer"          # legacy URL / session values

# ---- derived desalinated-water quantity ------------------------------------
if not np.isnan(d_val):
    st.caption(
        f"Desalinated water specified by **{spec_mode}** — calculated {d_name} = "
        f"**{d_val:.3f} {d_unit}**. All three stay consistent with each other."
    )

# ---- process train ---------------------------------------------------------
SLUG_TO_KEY = {s[3]: s[2] for s in STAGES}

# The URL seeds the selection ONCE, so a pasted or bookmarked ?unit= link opens
# on the stage it names. It is deliberately not re-read on later reruns: the
# tiles own the selection from then on and write the parameter back themselves,
# and re-reading it every pass would let a stale parameter overrule a click.
if "unit" not in st.session_state:
    st.session_state.unit = SLUG_TO_KEY.get(st.query_params.get("unit"), "Consumer")

# Alternating tile and arrow tracks, mirroring the 1fr / 30px gutter the train
# was drawn with when it was a single CSS grid.
_train_box = st.container(key="train")
with _train_box:
    _cols = st.columns([1, .11] * 4 + [1], vertical_alignment="center")

for i, (label, svg, key, slug, state) in enumerate(STAGES):
    if i:
        with _cols[2 * i - 1]:
            st.markdown(f'<div class="flow">{svg_flow()}</div>', unsafe_allow_html=True)
    col = cond_color(state)
    met = sum(1 for r in drink_report(state) if r[4] == "ok")
    on = " on" if st.session_state.unit == key else ""
    with _cols[2 * i]:
        st.markdown(
            f'<div class="stage{on}" style="--cond:{col}">'
            f'{svg}<div class="stage-name">{label}</div>'
            f'<div class="stage-si">SI <b>{fmt_signed(state["SI"])}</b></div>'
            f'<div class="stage-ccpp">{met}/{len(DRINK_CRITERIA)} drinking criteria</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Sits invisibly over the card above (see the .st-key-stagebtn- rules).
        # A button reruns the script in place; the anchor this replaced navigated
        # the browser, which discarded the session — doses, optimiser result and
        # the reader's sidebar all went with it on every single tile click.
        if st.button(f"Inspect {label}", key=f"stagebtn-{slug}"):
            st.session_state.unit = key
            st.query_params["unit"] = slug
            st.rerun()

st.caption("Select a unit to inspect the water at that point in the train.")

# ---- condition band + readout ---------------------------------------------
s = dict(zip([x[2] for x in STAGES], [x[4] for x in STAGES]))[st.session_state.unit]

# The verdict is whether this water can be supplied, so it is read off the
# drinking-water criteria. SI and CCPP stay as the headline figures — they are
# what the model computes — but neither of them decides the answer.
state_key = classify(s)
ccpp_now = ccpp_mg(s)
if state_key is None:
    verdict, sub, dot = ("Not available",
                         "The carbonate system has no solution here.", C_NEUTRAL)
else:
    verdict, sub = COND_TEXT[state_key]
    dot = COND_COLOR[state_key]

unit_label = KEY_TO_LABEL.get(st.session_state.unit, st.session_state.unit)
rows = drink_report(s)
chips = "".join(
    f'<span class="c {side}"><i>{label}</i><b>{fmt(v, dp)}</b><s>{unit}</s></span>'
    for label, v, unit, dp, side in rows
)

st.markdown(
    f'<div class="cond" style="--cond:{dot}">'
    f'  <div><div class="cond-k">saturation index</div>'
    f'       <div class="cond-v">{fmt_signed(s["SI"])}</div>'
    f'       <div class="cond-ccpp">CCPP <b>{fmt_signed(ccpp_now, 1)}</b> '
    f'mg/L as CaCO₃</div></div>'
    f'  <div><div class="cond-unit">{unit_label}</div>'
    f'       <div class="cond-txt">{verdict}</div>'
    f'       <div class="cond-sub">{sub}</div>'
    f'       <div class="crit">{chips}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)


def cells(items):
    html = ['<div class="readout">']
    for k, v, u in items:
        html.append(f'<div class="cell"><div class="k">{k}</div>'
                    f'<div class="v">{v}<span class="u">{u}</span></div></div>')
    html.append("</div>")
    return "".join(html)


# exactly eight values -> two flush rows of four, so the column rules align
readout = [
    ("pH", fmt(s["pH"]), ""),
    ("Alkalinity", fmt(s["Alk_mg"], 1), "mg/L as CaCO₃"),
    ("Ca²⁺", fmt(s["Ca_mg"], 1), "mg/L"),
    ("Mg²⁺", fmt(s["Mg_mg"], 1), "mg/L"),
    ("Total hardness", fmt(s["TH"], 1), "mg/L as CaCO₃"),
    ("Cₜ", fmt(s["CT"] * 1000, 3), "mmol/L"),
    ("CCPP", fmt_signed(ccpp_now, 1), "mg/L as CaCO₃"),
    ("Dissolved CO₂*", fmt(s["CO2"] * 1000, 3), "mmol/L"),
]
st.markdown(cells(readout), unsafe_allow_html=True)

# =============================================================================
# 11. AUTOMATIC STABILISATION
# =============================================================================
st.markdown("### Automatic stabilisation")
a1, a2 = st.columns([1, 2.4])
with a1:
    go_auto = st.button("Make the delivered water drinkable", type="primary",
                        width="stretch")
with a2:
    st.markdown(
        '<p class="note">Adjusts all four doses to bring every drinking-water '
        'criterion into range <b>at the consumer</b> — the end of the pipe, not '
        'the plant outlet. MgCl₂ moves too, because nothing else in the train '
        'supplies magnesium.</p>',
        unsafe_allow_html=True,
    )

if go_auto:
    with st.spinner("Searching for a drinkable operating point…"):
        result = auto_stabilize(base_inputs, {
            "CaOH2": float(st.session_state["dose_lime"]),
            "CO2": float(st.session_state["dose_co2"]),
            "CaCO3": float(st.session_state["dose_calcite"]),
            "MgCl2": float(st.session_state["dose_mgcl2"]),
        })
    o, fs = result["doses"], result["state"]
    st.session_state["pending_doses"] = {k: round(o[k], 1) for k in DOSE_VARS}
    misses = [r[0] for r in drink_report(fs) if r[4] != "ok"]
    st.session_state["auto_msg"] = (
        f'Ca(OH)₂ **{o["CaOH2"]:.1f}**, CO₂ **{o["CO2"]:.1f}**, CaCO₃ '
        f'**{o["CaCO3"]:.1f}**, MgCl₂ **{o["MgCl2"]:.1f}** mg/L → consumer pH '
        f'**{fs["pH"]:.2f}**, Ca **{fs["Ca_mg"]:.1f}**, Mg **{fs["Mg_mg"]:.1f}**, '
        f'alkalinity **{fs["Alk_mg"]:.0f}**, hardness **{fs["TH"]:.0f}** mg/L. '
        + result["message"]
        + (f' Still outside range: {", ".join(misses)}.' if misses else "")
    )
    st.session_state["auto_ok"] = bool(result["stable"])
    st.session_state["unit"] = "Consumer"
    st.query_params["unit"] = "consumer"
    st.rerun()

if "auto_msg" in st.session_state:
    (st.success if st.session_state.get("auto_ok") else st.warning)(st.session_state["auto_msg"])

# ---- single-dose suggestions (on demand: this one is expensive) ------------
with st.expander(f"Suggest a single-dose adjustment for {st.session_state.unit}"):
    st.markdown(
        '<p class="note">Scans one chemical at a time across its full range, holding '
        'everything else at the current settings, and reports the setting that brings '
        'the selected unit closest to meeting every drinking-water criterion.</p>',
        unsafe_allow_html=True)
    if st.button("Run scan", key="scan_btn"):
        target = st.session_state.unit
        with st.spinner("Scanning…"):
            def state_for(td):
                r = simulate(spec_mode, pH0, alk0, CT0, Ca0, Mg0, NACL_BACKGROUND, T0,
                             (td["CaOH2"], td["CO2"], td["CaCO3"], td["MgCl2"]),
                             t_res, Vw, Vhs, Tres, False, L, D, Q, Tenv, insulated,
                             res_steps=14, nseg=10, ccpp=False, profile=False)
                if r is None or r.get("problem"):
                    return None
                return {"Initial": r["initial"],
                        "Remineralization": r["remin"],
                        "Closed reservoir": r["res"]}.get(target, r["pipe"])

            base_q = quality(s)
            recs = []
            for chem in DOSE_VARS:
                best_v, best_q = doses[chem], base_q
                best_s = s
                for v in np.linspace(0, DOSE_MAX[chem], 25):
                    td = dict(doses)
                    td[chem] = float(v)
                    cs = state_for(td)
                    cq = quality(cs)
                    if cq < best_q:
                        best_q, best_v, best_s = cq, float(v), cs
                if base_q - best_q > 1e-4 and abs(best_v - doses[chem]) > 0.5:
                    recs.append((base_q - best_q, chem, best_v, best_s))
            recs.sort(reverse=True, key=lambda r: r[0])

        if is_stable(s):
            st.success(f"No adjustment needed — **{target}** already meets every "
                       f"drinking-water criterion.")
        elif recs:
            for _, chem, v, cs in recs[:3]:
                left = [r[0] for r in drink_report(cs) if r[4] != "ok"]
                st.markdown(
                    f"- **{CHEM_LABEL[chem]} → {v:.0f} mg/L** leaves "
                    + (f"{', '.join(left)} still out of range" if left
                       else "every criterion met")
                    + f" at {target}.")
        else:
            st.info("No single-dose change improves this unit. Try automatic stabilisation, "
                    "which moves all three together.")

# =============================================================================
# 12. ANALYSIS
# =============================================================================
st.divider()
st.markdown("### Analysis")

t1, t2, t3, t4, t5 = st.tabs([
    "Remineralization", "Reservoir", "Along the pipe", "Complete system",
    "Carbonate system",
])

with t1:
    c1, c2 = st.columns(2)
    chem = LABEL_CHEM[c1.selectbox("Chemical", list(CHEM_LABEL.values()))]
    out = c2.selectbox("Response", ["SI", "pH", "Alkalinity", "Total hardness", "CCPP"])
    xs = np.linspace(0, DOSE_MAX[chem], 40)
    need_ccpp = out == "CCPP"
    ys = []
    for x in xs:
        d = dict(doses)
        d[chem] = float(x)
        rr = remineralize(initial, d, need_ccpp)
        ys.append({"SI": rr["SI"], "pH": rr["pH"], "Alkalinity": rr["Alk_mg"],
                   "Total hardness": rr["TH"],
                   "CCPP": ccpp_mg(rr) if need_ccpp else np.nan}[out])
    ylab = {"SI": "SI", "pH": "pH", "Alkalinity": "Alkalinity [mg/L as CaCO₃]",
            "Total hardness": "Total hardness [mg/L as CaCO₃]",
            "CCPP": "CCPP [mg/L as CaCO₃]"}[out]
    fig = line_fig(xs, ys, f"{CHEM_LABEL[chem]} dose [mg/L]", ylab, si=(out == "SI"))
    fig.add_vline(x=doses[chem], line=dict(color=INK_MUTED, width=1.5, dash="dot"),
                  annotation_text="current", annotation_position="top",
                  annotation_font=dict(size=11, color=INK_MUTED))
    st.plotly_chart(fig, key="t1")
    st.caption("At the remineralization outlet. The other three doses stay at their "
               "current sidebar values.")

with t2:
    out2 = st.selectbox("Response", ["SI", "Dissolved CO₂", "pH", "Cₜ", "CCPP"],
                        key="res_out")
    ts = np.linspace(0, max(6.0, t_res * 2), 26)
    need_ccpp = out2 == "CCPP"
    ys = []
    for tt in ts:
        rr, _ = reservoir(remin, float(tt), Vw, Vhs, Tres, need_ccpp, 60, False)
        ys.append({"SI": rr["SI"], "pH": rr["pH"], "Cₜ": rr["CT"] * 1000,
                   "Dissolved CO₂": rr["CO2"] * 1000,
                   "CCPP": ccpp_mg(rr) if need_ccpp else np.nan}[out2])
    ylab2 = {"SI": "SI", "pH": "pH", "Cₜ": "Cₜ [mmol/L]",
             "Dissolved CO₂": "CO₂* [mmol/L]", "CCPP": "CCPP [mg/L as CaCO₃]"}[out2]
    fig = line_fig(ts, ys, "Residence time [h]", ylab2, si=(out2 == "SI"))
    fig.add_vline(x=t_res, line=dict(color=INK_MUTED, width=1.5, dash="dot"),
                  annotation_text="selected", annotation_position="top",
                  annotation_font=dict(size=11, color=INK_MUTED))
    st.plotly_chart(fig, key="t2")
    st.caption(
        "Sealed headspace: every mole crossing the interface changes pCO₂ in the gas, "
        f"so the two phases equilibrate and transfer stops. With {Vw:,.0f} m³ of water "
        f"over {Vhs:,.0f} m³ of headspace the gas is exhausted quickly, which is why "
        "the curve flattens and residence time then stops mattering.")

with t3:
    out3 = st.selectbox("Response", ["SI", "Cumulative CaCO₃ deposited",
                                     "Temperature", "pH"], key="pipe_out")
    col = {"SI": "SI", "Cumulative CaCO₃ deposited": "cum_precip",
           "Temperature": "T", "pH": "pH"}[out3]
    yy = profile[col] * (1000 if col == "cum_precip" else 1)
    ylab3 = {"SI": "SI", "Cumulative CaCO₃ deposited": "CaCO₃ deposited [mmol/L]",
             "Temperature": "Temperature [°C]", "pH": "pH"}[out3]
    fig = line_fig(profile["x"], yy, "Distance along pipe [m]", ylab3,
                   si=(out3 == "SI"))
    if out3 == "SI":
        pos = profile["SI"].to_numpy()
        if np.nanmax(pos) > 0:
            fig.add_trace(go.Scatter(
                x=profile["x"], y=np.where(pos > 0, pos, 0), mode="lines",
                line=dict(width=0), fill="tozeroy",
                fillcolor="rgba(194,65,12,.13)", hoverinfo="skip",
                showlegend=False))
    st.plotly_chart(fig, key="t3")
    st.caption(f"Water travels the {L:,.0f} m of the run from left to right. "
               "Shaded = oversaturated, where CaCO₃ deposits.")

with t4:
    out4 = st.selectbox("Parameter", ["SI", "CCPP", "pH", "Alkalinity", "Ca", "Mg",
                                      "Total hardness", "Cₜ"], key="sys_out")
    seq = [initial, remin, res, pipeout]
    names = ["Desalinated", "Remineralized", "Reservoir", "Consumer"]
    getter = {
        "SI": lambda z: z["SI"], "pH": lambda z: z["pH"],
        "Alkalinity": lambda z: z["Alk_mg"], "Ca": lambda z: z["Ca_mg"],
        "Mg": lambda z: z["Mg_mg"],
        "Total hardness": lambda z: z["TH"], "Cₜ": lambda z: z["CT"] * 1000,
        "CCPP": ccpp_mg,
    }[out4]
    ylab4 = {"SI": "SI", "pH": "pH", "Alkalinity": "Alkalinity [mg/L as CaCO₃]",
             "Ca": "Ca [mg/L]", "Mg": "Mg [mg/L]",
             "Total hardness": "Total hardness [mg/L as CaCO₃]",
             "Cₜ": "Cₜ [mmol/L]", "CCPP": "CCPP [mg/L as CaCO₃]"}[out4]
    # Marker colour describes the WATER at each stage, not the quantity plotted,
    # so it is shown whichever parameter is on the axis.
    st.plotly_chart(
        stage_figure(names, [getter(z) for z in seq], ylab4, si=(out4 == "SI"),
                     marker_colors=[cond_color(z) for z in seq]),
        key="t4")
    st.caption("Marker colour shows whether the water is fit to drink at each "
               "stage: blue = under-mineralized, green = meets every criterion, "
               "red = over-mineralized.")

with t5:
    st.plotly_chart(
        bjerrum_figure(s["T"], s["I"], s["pH"], [s["a0"], s["a1"], s["a2"]]),
        key="t5")
    st.caption(
        f"Distribution of total inorganic carbon across pH, at {s['T']:.0f} °C and the "
        f"ionic strength of this water ({s['I']*1000:.2f} mmol/L). Markers show "
        f"{st.session_state.unit}. Ionic strength shifts the crossover points — "
        "that shift is exactly what the activity correction accounts for."
    )

with st.expander("Full calculated state"):
    st.dataframe(pd.DataFrame([{
        "Stage": z["name"], "pH": z["pH"],
        "Alk [mg/L CaCO₃]": z["Alk_mg"], "Ca [mg/L]": z["Ca_mg"],
        "Mg [mg/L]": z["Mg_mg"], "TH [mg/L CaCO₃]": z["TH"],
        "Cₜ [mol/L]": z["CT"], "SI": z["SI"], "CCPP [mg/L CaCO₃]": ccpp_mg(z),
    } for z in (initial, remin, res, pipeout)]), hide_index=True)
