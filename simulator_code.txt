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

# Diverging axis for the saturation index: the one number this whole tool is
# about. Cool = the water dissolves carbonate, warm = it deposits it.
C_AGGR_STRONG = "#155E75"
C_AGGR = "#0891B2"
C_AGGR_SOFT = "#A5E8F5"
C_NEUTRAL = "#94A3B8"
C_SCALE_SOFT = "#FBC69B"
C_SCALE = "#C2410C"
C_SCALE_STRONG = "#7C2D12"
C_BALANCED = "#15803D"

# Teal / orange / green above are RESERVED: they encode saturation state and
# nothing else. Categorical series therefore draw from the remaining hue space,
# so a line on a chart can never be mistaken for a state. Validated all-pairs
# against the reserved three: normal-vision separation 16.8 worst case.
SERIES = ["#4F46E5", "#A21CAF", "#334E5C"]     # indigo, fuchsia, graphite
PRIMARY = "#334E5C"                            # buttons / neutral marks

INK = "#0B1F27"
INK_MID = "#4A6470"
INK_MUTED = "#5F7C87"
LINE = "#DCE7EC"
CARD = "#FFFFFF"

SI_COLORSCALE = [
    [0.00, C_AGGR_STRONG],
    [0.28, C_AGGR],
    [0.44, C_AGGR_SOFT],
    [0.50, "#EEF3F5"],
    [0.56, C_SCALE_SOFT],
    [0.72, C_SCALE],
    [1.00, C_SCALE_STRONG],
]

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
[data-testid="stSidebar"] {{ min-width: 306px; max-width: 306px; background: {CARD};
    border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2 {{ font-size: .74rem !important; text-transform: uppercase;
    letter-spacing: .1em; color: {INK_MUTED}; font-weight: 600; margin: .1rem 0 .3rem 0; }}
[data-testid="stSidebar"] label {{ font-size: .8rem !important; }}
[data-testid="stSidebar"] hr {{ margin: 1.05rem 0; }}

/* Streamlit's own vendor chrome has no place on a page being read as a product */
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}

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
.brand {{ display:flex; align-items:center; gap:1rem; }}
.brand svg {{ flex:none; }}
h1.wm {{ font-size: 2.15rem; font-weight: 700; line-height: 1.16; letter-spacing: -.03em;
    color: {INK}; margin: 0; padding: .06em 0 0 0; }}


/* ---- headline figures beside the title -------------------------------- */
.plant {{ display: flex; gap: 2.1rem; justify-content: flex-end; align-items: baseline;
    padding-top: .9rem; }}
.plant .fig {{ text-align: right; }}
.plant .fk {{ font-size: .68rem; font-weight: 600; letter-spacing: .085em;
    text-transform: uppercase; color: {INK_MUTED}; white-space: nowrap; }}
.plant .fv {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 1.72rem;
    font-weight: 600; letter-spacing: -.03em; line-height: 1.15;
    font-variant-numeric: tabular-nums; }}

/* ---- process train ---------------------------------------------------- */
.train {{ display: flex; align-items: stretch; gap: 0; margin: .2rem 0 0 0; }}
.stage {{ flex: 1 1 0; text-decoration: none; display: block; padding: .6rem .4rem .55rem .4rem;
    border: 1px solid {LINE}; background: #FAFCFD; position: relative;
    transition: background .2s ease, box-shadow .22s cubic-bezier(.2,.8,.3,1); }}
.stage:first-child {{ border-radius: 14px 0 0 14px; }}
.stage:last-child  {{ border-radius: 0 14px 14px 0; }}
.stage + .stage {{ border-left: none; }}
.stage svg {{ opacity: .62; transition: opacity .2s ease; }}
.stage:hover {{ background: {CARD}; z-index: 3; }}
.stage:hover svg {{ opacity: .88; }}
.stage.on {{ background: {CARD}; z-index: 4;
    box-shadow: inset 0 0 0 1.5px var(--cond), 0 14px 28px -18px rgba(11,31,39,.45); }}
.stage.on svg {{ opacity: 1; }}

/* Streamlit underlines and recolours markdown links; the stage tiles are
   navigation, not prose, so that styling is overridden outright. */
.stage, .stage:link, .stage:visited, .stage:hover, .stage:active,
.stage *, .train a, .train a * {{ text-decoration: none !important; }}
.stage-name {{ text-align:center; font-size:.83rem; font-weight:600;
    color:{INK} !important; margin-top:.1rem; letter-spacing:-.01em; }}
.stage-si {{ text-align:center; font-size:.9rem; color:{INK_MUTED} !important;
    margin-top:.2rem; font-family:'IBM Plex Mono',ui-monospace,monospace;
    font-variant-numeric: tabular-nums; }}
.stage-si b {{ color: var(--cond) !important; font-weight:600; }}

/* ---- hero reading ------------------------------------------------------ */
/* The saturation index is the one number this whole application computes, so
   it gets the focal point rather than being the tail of a sentence. */
.cond {{ display:grid; grid-template-columns: 168px 1fr; align-items:center;
    gap:1.1rem; padding:.9rem 1.2rem; border:1px solid {LINE};
    border-radius:13px; background:{CARD}; margin:.1rem 0 1rem 0; }}
.cond-k {{ font-size:.68rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
    color:{INK_MUTED}; }}
.cond-v {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:3.05rem;
    font-weight:600; line-height:1.02; letter-spacing:-.04em; color:var(--cond);
    font-variant-numeric: tabular-nums; }}
.cond-unit {{ font-size:1.28rem; font-weight:600; color:{INK}; letter-spacing:-.02em; }}
.cond-txt {{ font-size:.97rem; color:{INK}; font-weight:600; margin-top:.12rem; }}
.cond-sub {{ font-size:.86rem; color:{INK_MID}; font-weight:400; margin-top:.1rem;
    max-width: 46ch; }}

/* ---- instrument readout ----------------------------------------------- */
/* Fixed six-track module: 12 values fill exactly two flush rows, so the
   vertical rules line up and the card closes cleanly. */
.readout {{ display:grid; grid-template-columns:repeat(6, 1fr);
    border:1px solid {LINE}; border-radius:13px; background:{CARD}; overflow:hidden;
    animation: rise .32s cubic-bezier(.2,.8,.3,1); }}
@media (max-width: 1250px) {{ .readout {{ grid-template-columns:repeat(4, 1fr); }} }}
@media (max-width: 820px)  {{ .readout {{ grid-template-columns:repeat(2, 1fr); }} }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(5px); }}
                   to {{ opacity:1; transform:none; }} }}
.cell {{ padding:.62rem .8rem .66rem .8rem; border-right:1px solid {LINE};
    border-bottom:1px solid {LINE}; }}
.cell.hl {{ background:#FBFDFE; }}

/* unit-specific metrics: one row, so it never fights the grid module above */
.extras {{ display:flex; flex-wrap:wrap; margin:.5rem 0 0 0; border:1px solid {LINE};
    border-radius:13px; background:{CARD}; overflow:hidden; }}
.extras .ex {{ flex:1 1 0; min-width:150px; padding:.55rem .85rem .6rem .85rem;
    border-right:1px solid {LINE}; }}
.extras .ex:last-child {{ border-right:none; }}
.extras .k {{ font-size:.7rem; font-weight:600; letter-spacing:.03em; color:{INK_MUTED}; }}
.extras .v {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:1rem;
    font-weight:600; color:{INK}; margin-top:.14rem; font-variant-numeric:tabular-nums; }}
.extras .u {{ font-family:'IBM Plex Sans', sans-serif; font-size:.7rem; color:{INK_MUTED};
    font-weight:500; margin-left:.2rem; }}
/* Deliberately NOT uppercased: these labels carry element symbols and Greek
   letters (Ca²⁺, Mg²⁺, pH, γ₂) that uppercasing would render incorrectly. */
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
.ionrow {{ display:flex; align-items:center; gap:1.1rem; margin:.2rem 0 .8rem 0; }}
.ionrow svg {{ flex:none; }}
.ioncap {{ font-size:.86rem; color:{INK_MID}; line-height:1.55; max-width:66ch; }}
.charttitle {{ font-size:.72rem; font-weight:600; letter-spacing:.085em; text-transform:uppercase;
    color:{INK_MUTED}; margin:.7rem 0 .35rem 0; }}
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

STABLE_SI_TOL = 0.02
STABLE_PH_MIN, STABLE_PH_MAX = 7.0, 8.5

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
    """Consumer state alone, at reduced resolution — for scans and optimisation."""
    r = simulate(mode, pH0, alk0, CT0, Ca0, Mg0, nacl, T0,
                 (doses["CaOH2"], doses["CO2"], doses["CaCO3"], doses["MgCl2"]),
                 t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated,
                 res_steps=14 if fast else 60, nseg=10 if fast else 80,
                 ccpp=False, profile=False)
    return None if r is None or r.get("problem") else r["pipe"]


# =============================================================================
# 6. AUTOMATIC STABILISATION
# =============================================================================
class StablePointFound(Exception):
    def __init__(self, x, state):
        self.x = np.asarray(x, dtype=float)
        self.state = state


def is_stable(s):
    if s is None or np.isnan(s["SI"]) or np.isnan(s["pH"]):
        return False
    return abs(s["SI"]) <= STABLE_SI_TOL and STABLE_PH_MIN <= s["pH"] <= STABLE_PH_MAX


def quality(s):
    """Lower is better: distance from SI = 0 plus a penalty outside the pH window."""
    if s is None or np.isnan(s["SI"]) or np.isnan(s["pH"]):
        return np.inf
    q = abs(s["SI"])
    if s["pH"] < STABLE_PH_MIN:
        q += 0.5 * (STABLE_PH_MIN - s["pH"])
    elif s["pH"] > STABLE_PH_MAX:
        q += 0.5 * (s["pH"] - STABLE_PH_MAX)
    return float(q)


def auto_stabilize(base, current, fast_first=True):
    """Search Ca(OH)2 / CO2 / CaCO3 for SI = 0 at the consumer.

    Powell with a hard evaluation cap, an early exit as soon as the target box
    is reached, then a full-resolution validation of whatever it found.
    """
    def evaluate(x, fast):
        d = {"CaOH2": float(x[0]), "CO2": float(x[1]), "CaCO3": float(x[2]),
             "MgCl2": current["MgCl2"]}
        return consumer_only(**base, doses=d, fast=fast), d

    def objective(x, fast):
        s, _ = evaluate(x, fast)
        if s is None or np.isnan(s["SI"]) or np.isnan(s["pH"]):
            return 1e6
        if is_stable(s):
            raise StablePointFound(x, s)
        score = (s["SI"] / 0.10) ** 2
        if s["pH"] < STABLE_PH_MIN:
            score += 12.0 * (STABLE_PH_MIN - s["pH"]) ** 2
        elif s["pH"] > STABLE_PH_MAX:
            score += 12.0 * (s["pH"] - STABLE_PH_MAX) ** 2
        ref = np.array([current["CaOH2"], current["CO2"], current["CaCO3"]])
        score += 0.015 * np.sum(((np.asarray(x) - ref) / np.array([200., 200., 300.])) ** 2)
        return float(score)

    bounds = [(0.0, DOSE_MAX["CaOH2"]), (0.0, DOSE_MAX["CO2"]), (0.0, DOSE_MAX["CaCO3"])]
    x0 = np.array([current["CaOH2"], current["CO2"], current["CaCO3"]], dtype=float)

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

    x_fast = powell(x0, fast_first, 120)
    cand = [(current, now)]
    for x in (x_fast, powell(x_fast, False, 60)):
        d = {"CaOH2": float(np.clip(x[0], 0, DOSE_MAX["CaOH2"])),
             "CO2": float(np.clip(x[1], 0, DOSE_MAX["CO2"])),
             "CaCO3": float(np.clip(x[2], 0, DOSE_MAX["CaCO3"])),
             "MgCl2": current["MgCl2"]}
        cand.append((d, consumer_only(**base, doses=d, fast=False)))

    best_d, best_s = min(cand, key=lambda p: quality(p[1]))
    return {"doses": dict(best_d), "state": best_s, "stable": is_stable(best_s),
            "message": ("A stable operating point was found." if is_stable(best_s)
                        else "No point inside the target box was found within the "
                             "evaluation limit; the closest one is shown.")}


# =============================================================================
# 7. FIGURES
# =============================================================================
def si_color(si):
    """Discrete state colour: aggressive / at equilibrium / scaling."""
    if si is None or np.isnan(si):
        return C_NEUTRAL
    if abs(si) <= STABLE_SI_TOL:
        return C_BALANCED
    return C_SCALE if si > 0 else C_AGGR


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


def stage_figure(names, values, ytitle, si=False):
    colors = [si_color(v) for v in values] if si else [SERIES[0]] * len(values)
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
def svg_calcite(size=54):
    """The calcite cleavage rhombohedron — the mineral this whole app is about.

    Drawn on the real crystallography, not a generic cube. Calcite is trigonal
    R3̄c; its {101̄4} cleavage rhomb has face angles of 78°05′ / 101°55′ and
    interfacial angles of 74°55′ / 105°05′. The vertex coordinates below are the
    orthographic projection of that solid down a mirror-symmetric azimuth, so
    the silhouette is a true calcite rhomb rather than a squashed hexagon.

    (The common error is drawing the faces at 74°55′ — that is the dihedral
    angle between two faces in three dimensions, not the angle you see.)
    """
    return f"""<svg width="{size}" height="{size * 100 / 135.08:.1f}"
      viewBox="0 0 135.08 100" fill="none" preserveAspectRatio="xMidYMid meet">
      <polygon points="67.54,100 135.08,86.16 135.08,13.84 67.54,0 0,13.84 0,86.16"
               fill="#F0F7FA" stroke="{INK}" stroke-width="3.2" stroke-linejoin="round"/>
      <polygon points="67.54,72.33 135.08,86.16 135.08,13.84 67.54,0"
               fill="{C_AGGR_SOFT}" fill-opacity=".55"/>
      <polygon points="67.54,72.33 67.54,0 0,13.84 0,86.16"
               fill="{C_SCALE_SOFT}" fill-opacity=".45"/>
      <path d="M67.54 72.33 L135.08 86.16 M67.54 72.33 L0 86.16 M67.54 72.33 L67.54 0"
            stroke="{INK}" stroke-width="2.6" stroke-linecap="round"/>
      <polygon points="67.54,100 135.08,86.16 67.54,72.33 0,86.16"
               fill="{CARD}" fill-opacity=".35"/>
    </svg>"""


def svg_carbonate(size=46):
    """The carbonate ion, CO₃²⁻: trigonal planar, D₃ₕ, three identical C–O bonds
    at exactly 120°. Drawn as the delocalised hybrid (equal bonds plus an inner
    arc) rather than one of the three resonance structures, because the real ion
    has a bond order of 1⅓ on every bond and −⅔ charge on every oxygen."""
    import math as _m
    cx = cy = 50.0
    r = 30.0
    pts = [(cx + r * _m.cos(_m.radians(a)), cy + r * _m.sin(_m.radians(a)))
           for a in (270, 30, 150)]                     # point-up, 120° apart
    bonds = "".join(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" '
                    f'stroke="{INK}" stroke-width="3"/>' for x, y in pts)
    oxy = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="11" fill="{CARD}" '
                  f'stroke="{INK}" stroke-width="3"/>' for x, y in pts)
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none">
      {bonds}
      <circle cx="{cx}" cy="{cy}" r="16.5" fill="none" stroke="{C_AGGR}"
              stroke-width="2.4" stroke-dasharray="4 3.4"/>
      {oxy}
    </svg>"""


# ---------------------------------------------------------------------------
# Process-unit symbols, drawn on ISO 10628-2 geometry rather than as free-hand
# icons. The standard is built on a 2.5 mm module; here 1 ISO mm = 4 SVG units,
# so the module is 10 units and every dimension lands on an integer.
#
# Line-weight hierarchy is the standard's own 4:2:1 family (ISO 128-20):
#     process piping 2.0  >  equipment outline 1.4  >  fittings/detail 1.0
# Piping is deliberately HEAVIER than equipment: the pipe network is the
# subject of a process drawing, vessels are context. Symbols are unfilled
# line art - in the ISO sheets ~96% of elements carry fill:none, and solid
# black is reserved for small marks that mean something (flow direction,
# valve seats). Flow reads left to right; turns are 90 degrees only.
# ---------------------------------------------------------------------------
W_PIPE, W_EQUIP, W_DETAIL = 2.0, 1.4, 1.0
SVG_OPEN = ('<svg width="100%" height="88" viewBox="0 0 440 210" fill="none" '
            'preserveAspectRatio="xMidYMid meet" '
            'stroke-linecap="round" stroke-linejoin="round">')


def _arrow(x, y, d=1):
    """ISO flow arrow: a solid triangle at 2:1 length-to-base."""
    L, B = 15.0, 7.5
    return (f'<polygon points="{x:.0f},{y-B:.0f} {x + d*L:.0f},{y:.0f} '
            f'{x:.0f},{y+B:.0f}" fill="{INK}"/>')


def svg_desal():
    """Reverse-osmosis pressure vessel.

    ISO has no membrane symbol, so this follows water-industry practice built
    from ISO primitives: a horizontal pressure vessel with 2:1 semi-elliptical
    heads (rx = D/4 - the deeper dish is what reads as *pressure*-rated rather
    than an atmospheric tank), and a dashed lengthwise barrier offset above the
    centreline. Feed and concentrate connect on one side of that barrier, the
    permeate leaves from the other - that relationship is what makes it an RO
    vessel and not a filter.
    """
    return f"""{SVG_OPEN}
    <path d="M90 60 H350 M90 150 H350" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M90 150 A30 45 0 0 1 90 60" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M350 60 A30 45 0 0 1 350 150" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M90 95 H350" stroke="{INK}" stroke-width="{W_EQUIP}" stroke-dasharray="8 8"/>
    <path d="M20 125 H60" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(58, 125)}
    <path d="M380 125 H420" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(398, 125)}
    <path d="M300 60 V22" stroke="{INK}" stroke-width="{W_PIPE}"/>
    <path d="M292 32 l8 -10 l8 10" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    </svg>"""


def svg_remin():
    """Chemical dosing station: day tank, metering pump, injection quill.

    The pump is ISO X8095 - a circle with a discharge chevron and, on the left,
    an outward-bulging arc of radius r*sqrt(2). That arc is the diaphragm, and
    it is the only thing distinguishing a positive-displacement dosing pump
    from a centrifugal one. Chemical additions enter a process line from above
    by convention, so the station sits over the pipe and injects downward.
    """
    return f"""{SVG_OPEN}
    <path d="M150 14 H230 V96 H150 Z" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M150 34 H230" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M180 24 L190 34 L200 24" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    <path d="M190 96 V116" stroke="{INK}" stroke-width="{W_PIPE}"/>
    <circle cx="190" cy="140" r="24" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M190 116 L214 140 L190 164" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M166 140 H214" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M190 164 A34 34 0 0 1 190 116" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M214 140 H262 V178" stroke="{INK}" stroke-width="{W_PIPE}"/>
    <path d="M252 178 l10 10 l10 -10" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    <path d="M20 190 H420" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(360, 190)}
    </svg>"""


def svg_res():
    """Storage tank with vapour headspace and vent.

    ISO X2063 conical roof at a rise of D/8, and the single horizontal line at
    20% depth that denotes the liquid surface - in P&ID the headspace is that
    one line, never a shaded or gradient-filled region. The vent (ISO 2039) is
    an OPEN chevron on a stem: an open chevron is the standard's vocabulary for
    "to or from atmosphere", which is exactly what the vented mode models.
    """
    return f"""{SVG_OPEN}
    <path d="M140 46 L220 26 L300 46" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M140 46 V186 H300 V46" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M140 82 H300" stroke="{INK}" stroke-width="{W_EQUIP}"/>
    <path d="M200 62 L220 82 L240 62 M240 62 H268" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    <path d="M200 26 L220 6 L240 26" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    <path d="M220 26 V6" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    <path d="M20 108 H140" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(104, 108)}
    <path d="M300 160 H420" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(384, 160)}
    </svg>"""


def svg_pipe():
    """Insulated supply run, ISO X322.

    The process line at full weight with two thin lines offset by D/8 either
    side and 45-degree hatching spanning the gap - the standard's insulation
    band. Flow arrows are placed at entry, midpoint and exit, as a real drawing
    does, not once per line.
    """
    hatch = "".join(f'<path d="M{x} 130 l20 -20" stroke="{INK}" stroke-width="{W_DETAIL}"/>'
                    for x in range(60, 380, 26))
    return f"""{SVG_OPEN}
    <path d="M20 120 H420" stroke="{INK}" stroke-width="{W_PIPE}"/>
    <path d="M50 100 H390 M50 130 H390" stroke="{INK}" stroke-width="{W_DETAIL}"/>
    {hatch}
    {_arrow(120, 120)}{_arrow(280, 120)}
    </svg>"""


def svg_consumer():
    """Off-page connector - the standard way a P&ID hands a stream onward.

    A pennant: a square tail the pipe attaches to, stepping out and tapering to
    a point in the flow direction. This is the most-used connector in real
    drawings, and it is the correct symbol for "leaves this diagram and enters
    the distribution network".
    """
    return f"""{SVG_OPEN}
    <path d="M20 105 H190" stroke="{INK}" stroke-width="{W_PIPE}"/>
    {_arrow(150, 105)}
    <path d="M220 125 V85 H260 V65 L300 105 L260 145 V125 Z"
          stroke="{INK}" stroke-width="{W_DETAIL}"/>
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

with st.sidebar:
    st.header("Feed water")
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
    nacl0 = st.number_input("Background salinity [mg/L as NaCl]", 0.0, 2000.0, 60.0, 5.0,
                            help="Residual permeate salts. Sets the chloride level; "
                                 "sodium follows from electroneutrality. Feeds the "
                                 "ionic-strength correction.")
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
    vented = st.toggle(
        "Vented to atmosphere", False,
        help="Closed: a sealed headspace of the volume above. It equilibrates with "
             "the water almost immediately, so residence time then stops mattering. "
             "Vented: open to air at ambient pCO₂, an effectively infinite reservoir, "
             "so CO₂ transfer continues for the whole residence time.")
    Vhs = st.number_input("Headspace volume [m³]", 0.01, 1e7, 100.0, 10.0,
                          disabled=vented,
                          help=("Not used when the reservoir is vented — the "
                                "atmosphere is unbounded." if vented else None))
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
            f"""
- kLa = `{KLA:.1e}` s⁻¹ · k_s = `{KS_SCALE:.1e}` mol L⁻¹s⁻¹ per unit SI
- U = `{U_EXPOSED:.0f}` (exposed) / `{U_INSULATED:.0f}` (insulated) W m⁻²K⁻¹
- Ambient / initial headspace pCO₂ = `{PCO2_GAS_0:.2e}` atm
- Reservoir gas phase: **finite sealed headspace** (closed) or **unbounded
  atmosphere at fixed pCO₂** (vented)
- K₁, K₂, K_w, K_sp, K_H temperature-corrected with Van't Hoff
- Activity coefficients from the **Davies equation**, applied as conditional
  constants and iterated with the ionic strength
- pH reported on the **activity scale**; SI built from **ion activities**
"""
        )

# =============================================================================
# 10. RUN
# =============================================================================
doses = {"CaOH2": d_lime, "CO2": d_co2, "CaCO3": d_calc, "MgCl2": d_mg}
base_inputs = dict(mode=spec_mode, pH0=pH0, alk0=alk0, CT0=CT0, Ca0=Ca0, Mg0=Mg0,
                   nacl=nacl0, T0=T0, t_res=t_res, Vw=Vw, Vhs=Vhs, Tres=Tres,
                   vented=vented, L=L, D=D, Q=Q, Tenv=Tenv, insulated=insulated)

sim = simulate(spec_mode, pH0, alk0, CT0, Ca0, Mg0, nacl0, T0,
               (d_lime, d_co2, d_calc, d_mg),
               t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated)

HEAD_LEDE = (
    '<p class="lede">Desalinated permeate is aggressive: almost no calcium, almost no '
    'buffering. This tool doses it back to stability and follows the calcium-carbonate '
    'balance through storage and kilometres of pipe — where temperature, CO₂ exchange '
    'and residence time keep moving it.</p>'
)
HEAD_BRAND = (f'<div class="brand">{svg_calcite(50)}'
              f'<div><h1 class="wm">Remineralization<br>&amp; water transport</h1></div></div>')

if sim is None or sim.get("problem"):
    st.markdown(HEAD_BRAND + HEAD_LEDE, unsafe_allow_html=True)
    st.error(sim["problem"] if sim else "This feed water cannot be solved.")
    st.stop()

initial, remin, res, pipeout = sim["initial"], sim["remin"], sim["res"], sim["pipe"]
profile, res_extra, pipe_extra = sim["profile"], sim["res_x"], sim["pipe_x"]
d_name, d_unit, d_val = sim["derived"]

# Title on the left; the three figures that summarise the whole run on the right,
# so the first fold answers "what is coming out of the tap" before any scrolling.
head_l, head_r = st.columns([1.55, 1])
with head_l:
    st.markdown(HEAD_BRAND + HEAD_LEDE, unsafe_allow_html=True)
with head_r:
    st.markdown(
        f'<div class="plant">'
        f'  <div class="fig"><div class="fk">delivered SI</div>'
        f'       <div class="fv" style="color:{si_color(pipeout["SI"])}">'
        f'{fmt_signed(pipeout["SI"])}</div></div>'
        f'  <div class="fig"><div class="fk">delivered pH</div>'
        f'       <div class="fv" style="color:{INK}">{fmt(pipeout["pH"])}</div></div>'
        f'  <div class="fig"><div class="fk">CaCO₃ laid down</div>'
        f'       <div class="fv" style="color:{INK}">{fmt(pipe_extra["precip"]*1000, 3)}'
        f'<span style="font-size:.78rem;color:{INK_MUTED}"> mmol/L</span></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Four stages, because the model computes four distinct states. The supply pipe
# is not a fifth state — it is the transport BETWEEN the reservoir and the tap,
# and its own metrics (velocity, travel time, deposition) appear on the outlet.
# Showing it as a separate tile reporting identical numbers to the Consumer, as
# an earlier layout did, invites the reader to click and find nothing changed.
STAGES = [
    ("Desalinated water", svg_desal(), "Initial", "initial", initial),
    ("Remineralization", svg_remin(), "Remineralization", "remineralization", remin),
    ("Vented reservoir" if vented else "Closed reservoir",
     svg_res(), "Closed reservoir", "reservoir", res),
    ("Supply pipe", svg_pipe(), "Supply pipe", "pipe", pipeout),
    ("Consumer", svg_consumer(), "Consumer", "consumer", pipeout),
]
KEY_TO_LABEL = {k: lbl for lbl, _, k, _, _ in STAGES}
KEY_TO_LABEL["Supply pipe"] = "Consumer"          # legacy URL / session values

# ---- derived feed-water quantity ------------------------------------------
if not np.isnan(d_val):
    st.caption(
        f"Feed water specified by **{spec_mode}** — calculated {d_name} = "
        f"**{d_val:.3f} {d_unit}**. All three stay consistent with each other."
    )

# ---- process train ---------------------------------------------------------
if "unit" not in st.session_state:
    st.session_state.unit = "Consumer"
url_unit = st.query_params.get("unit")
slug_to_key = {s[3]: s[2] for s in STAGES}
if url_unit in slug_to_key:
    st.session_state.unit = slug_to_key[url_unit]

train = ['<div class="train">']
for label, svg, key, slug, state in STAGES:
    si = state["SI"]
    col = si_color(si)
    on = " on" if st.session_state.unit == key else ""
    train.append(
        f'<a class="stage{on}" href="?unit={slug}" target="_self" style="--cond:{col}">'
        f'{svg}<div class="stage-name">{label}</div>'
        f'<div class="stage-si">SI <b>{fmt_signed(si)}</b></div></a>'
    )
train.append("</div>")
st.markdown("".join(train), unsafe_allow_html=True)
st.caption("Select a unit to inspect the water at that point in the train.")

# ---- condition band + readout ---------------------------------------------
s = dict(zip([x[2] for x in STAGES], [x[4] for x in STAGES]))[st.session_state.unit]

if np.isnan(s["SI"]):
    verdict, sub, dot = "Not available", "The carbonate system has no solution here.", C_NEUTRAL
elif abs(s["SI"]) <= STABLE_SI_TOL:
    verdict, sub, dot = ("At calcite equilibrium",
                         "The water neither dissolves nor deposits CaCO₃.", C_BALANCED)
elif s["SI"] > 0:
    verdict, sub, dot = ("Oversaturated — scaling tendency",
                         f"CaCO₃ tends to deposit. SI = {s['SI']:+.2f}.", C_SCALE)
else:
    verdict, sub, dot = ("Undersaturated — aggressive",
                         f"The water dissolves CaCO₃ and attacks cement lining. "
                         f"SI = {s['SI']:+.2f}.", C_AGGR)

unit_label = KEY_TO_LABEL.get(st.session_state.unit, st.session_state.unit)
si_txt = fmt_signed(s["SI"])

st.markdown(
    f'<div class="cond" style="--cond:{dot}">'
    f'  <div><div class="cond-k">saturation index</div>'
    f'       <div class="cond-v">{si_txt}</div></div>'
    f'  <div><div class="cond-unit">{unit_label}</div>'
    f'       <div class="cond-txt">{verdict}</div>'
    f'       <div class="cond-sub">{sub}</div></div>'
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


# exactly twelve values -> two flush rows of six, so the column rules align
readout = [
    ("pH", fmt(s["pH"]), ""),
    ("Alkalinity", fmt(s["Alk_mg"], 1), "mg/L as CaCO₃"),
    ("Ca²⁺", fmt(s["Ca_mg"], 1), "mg/L"),
    ("Mg²⁺", fmt(s["Mg_mg"], 1), "mg/L"),
    ("Total hardness", fmt(s["TH"], 1), "mg/L as CaCO₃"),
    ("Cₜ", fmt(s["CT"] * 1000, 3), "mmol/L"),
    ("CCPP", fmt(s.get("CCPP", np.nan) * 1000 if s.get("CCPP") is not None else np.nan, 3),
     "mmol/L"),
    ("Dissolved CO₂*", fmt(s["CO2"] * 1000, 3), "mmol/L"),
    ("Ionic strength", fmt(s["I"] * 1000, 2), "mmol/L"),
    ("γ₂ activity coeff.", fmt(s["g2"], 3), ""),
    ("Chloride", fmt(s["Cl"] * MW["Cl"] * 1000, 1), "mg/L"),
    ("Temperature", fmt(s["T"], 1), "°C"),
]
st.markdown(cells(readout), unsafe_allow_html=True)

# unit-specific metrics live in their own strip rather than a second grid whose
# column module would not line up with the one above it
extra = []
if st.session_state.unit == "Closed reservoir":
    extra = [("Headspace pCO₂", f'{res_extra["pCO2"]:.2e}', "atm"),
             ("Residence time", fmt(t_res, 2), "h"),
             ("Gas phase", "vented" if vented else "sealed", "")]
elif st.session_state.unit in ("Supply pipe", "Consumer"):
    extra = [("Velocity", fmt(pipe_extra["v"]), "m/s"),
             ("Travel time", fmt(pipe_extra["t_s"] / 60), "min"),
             ("Heat transfer U", fmt(pipe_extra["U"]), "W/m²K"),
             ("CaCO₃ deposited in pipe", fmt(pipe_extra["precip"] * 1000, 4), "mmol/L")]
if extra:
    strip = ['<div class="extras">']
    for k, v, u in extra:
        strip.append(f'<div class="ex"><div class="k">{k}</div>'
                     f'<div class="v">{v}<span class="u">{u}</span></div></div>')
    strip.append("</div>")
    st.markdown("".join(strip), unsafe_allow_html=True)

# =============================================================================
# 11. AUTOMATIC STABILISATION
# =============================================================================
st.markdown("### Automatic stabilisation")
a1, a2 = st.columns([1, 2.4])
with a1:
    go_auto = st.button("Stabilise the delivered water", type="primary",
                        width="stretch")
with a2:
    st.markdown(
        f'<p class="note">Adjusts Ca(OH)₂, CO₂ and CaCO₃ only, targeting '
        f'|SI| ≤ {STABLE_SI_TOL:.2f} and pH {STABLE_PH_MIN:.1f}–{STABLE_PH_MAX:.1f} '
        f'<b>at the consumer</b> — the end of the pipe, not the plant outlet.</p>',
        unsafe_allow_html=True,
    )

if go_auto:
    with st.spinner("Searching for a stable operating point…"):
        result = auto_stabilize(base_inputs, {
            "CaOH2": float(st.session_state["dose_lime"]),
            "CO2": float(st.session_state["dose_co2"]),
            "CaCO3": float(st.session_state["dose_calcite"]),
            "MgCl2": float(st.session_state["dose_mgcl2"]),
        })
    o, fs = result["doses"], result["state"]
    st.session_state["pending_doses"] = {k: round(o[k], 1) for k in ("CaOH2", "CO2", "CaCO3")}
    st.session_state["auto_msg"] = (
        f'Ca(OH)₂ **{o["CaOH2"]:.1f}**, CO₂ **{o["CO2"]:.1f}**, CaCO₃ **{o["CaCO3"]:.1f}** mg/L '
        f'→ consumer SI **{fs["SI"]:+.2f}**, pH **{fs["pH"]:.2f}**. {result["message"]}'
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
        'the selected unit closest to equilibrium.</p>', unsafe_allow_html=True)
    if st.button("Run scan", key="scan_btn"):
        target = st.session_state.unit
        with st.spinner("Scanning…"):
            def state_for(td):
                r = simulate(spec_mode, pH0, alk0, CT0, Ca0, Mg0, nacl0, T0,
                             (td["CaOH2"], td["CO2"], td["CaCO3"], td["MgCl2"]),
                             t_res, Vw, Vhs, Tres, vented, L, D, Q, Tenv, insulated,
                             res_steps=14, nseg=10, ccpp=False, profile=False)
                if r is None:
                    return None
                return {"Initial": r["initial"], "Remineralization": r["remin"],
                        "Closed reservoir": r["res"]}.get(target, r["pipe"])

            base_q = quality(s)
            recs = []
            for chem in ("CaOH2", "CO2", "CaCO3"):
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
            st.success(f"No adjustment needed — **{target}** already meets the target.")
        elif recs:
            for _, chem, v, cs in recs[:3]:
                st.markdown(
                    f"- **{CHEM_LABEL[chem]} → {v:.0f} mg/L** gives SI **{cs['SI']:+.2f}**, "
                    f"pH **{cs['pH']:.2f}** at {target}.")
        else:
            st.info("No single-dose change improves this unit. Try automatic stabilisation, "
                    "which moves all three together.")

# =============================================================================
# 12. ANALYSIS
# =============================================================================
st.divider()
st.markdown("### Analysis")

t1, t2, t3, t4, t5, t6 = st.tabs([
    "Dose response", "Reservoir", "Along the pipe", "Across the train",
    "Carbonate system", "Operating map",
])

with t1:
    c1, c2 = st.columns(2)
    chem = LABEL_CHEM[c1.selectbox("Chemical", list(CHEM_LABEL.values()))]
    out = c2.selectbox("Response", ["SI", "pH", "Alkalinity", "Total hardness",
                                    "CCPP", "Ionic strength"])
    xs = np.linspace(0, DOSE_MAX[chem], 40)
    need_ccpp = out == "CCPP"
    ys = []
    for x in xs:
        d = dict(doses)
        d[chem] = float(x)
        rr = remineralize(initial, d, need_ccpp)
        ys.append({"SI": rr["SI"], "pH": rr["pH"], "Alkalinity": rr["Alk_mg"],
                   "Total hardness": rr["TH"],
                   "CCPP": rr.get("CCPP", np.nan) * 1000 if need_ccpp else np.nan,
                   "Ionic strength": rr["I"] * 1000}[out])
    ylab = {"SI": "SI", "pH": "pH", "Alkalinity": "Alkalinity [mg/L as CaCO₃]",
            "Total hardness": "Total hardness [mg/L as CaCO₃]",
            "CCPP": "CCPP [mmol/L]", "Ionic strength": "Ionic strength [mmol/L]"}[out]
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
        rr, _ = reservoir(remin, float(tt), Vw, Vhs, Tres, need_ccpp, 60, vented)
        ys.append({"SI": rr["SI"], "pH": rr["pH"], "Cₜ": rr["CT"] * 1000,
                   "Dissolved CO₂": rr["CO2"] * 1000,
                   "CCPP": rr.get("CCPP", np.nan) * 1000 if need_ccpp else np.nan}[out2])
    ylab2 = {"SI": "SI", "pH": "pH", "Cₜ": "Cₜ [mmol/L]",
             "Dissolved CO₂": "CO₂* [mmol/L]", "CCPP": "CCPP [mmol/L]"}[out2]
    fig = line_fig(ts, ys, "Residence time [h]", ylab2, si=(out2 == "SI"))
    fig.add_vline(x=t_res, line=dict(color=INK_MUTED, width=1.5, dash="dot"),
                  annotation_text="selected", annotation_position="top",
                  annotation_font=dict(size=11, color=INK_MUTED))
    st.plotly_chart(fig, key="t2")
    if vented:
        st.caption(
            "Open to atmosphere: the air is an unbounded reservoir at ambient pCO₂, so "
            "exchange continues for the whole residence time. Storage alone can do much "
            "of the stabilisation work here.")
    else:
        st.caption(
            "Sealed headspace: every mole crossing the interface changes pCO₂ in the gas, "
            f"so the two phases equilibrate and transfer stops. With {Vw:,.0f} m³ of water "
            f"over {Vhs:,.0f} m³ of headspace the gas is exhausted quickly, which is why "
            "the curve flattens and residence time then stops mattering. Switch on "
            "**Vented to atmosphere** in the sidebar to see the unbounded case.")

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
    st.caption(f"Water travels {L:,.0f} m in {pipe_extra['t_s']/60:.1f} min at "
               f"{pipe_extra['v']:.2f} m/s. Shaded = oversaturated, where CaCO₃ deposits.")

with t4:
    out4 = st.selectbox("Parameter", ["SI", "pH", "Alkalinity", "Ca", "Total hardness",
                                      "Cₜ", "CCPP", "Ionic strength", "Temperature"],
                        key="sys_out")
    seq = [initial, remin, res, pipeout]
    names = ["Desalinated", "Remineralized", "Reservoir", "Consumer"]
    getter = {
        "SI": lambda z: z["SI"], "pH": lambda z: z["pH"],
        "Alkalinity": lambda z: z["Alk_mg"], "Ca": lambda z: z["Ca_mg"],
        "Total hardness": lambda z: z["TH"], "Cₜ": lambda z: z["CT"] * 1000,
        "CCPP": lambda z: z.get("CCPP", np.nan) * 1000,
        "Ionic strength": lambda z: z["I"] * 1000, "Temperature": lambda z: z["T"],
    }[out4]
    ylab4 = {"SI": "SI", "pH": "pH", "Alkalinity": "Alkalinity [mg/L as CaCO₃]",
             "Ca": "Ca [mg/L]", "Total hardness": "Total hardness [mg/L as CaCO₃]",
             "Cₜ": "Cₜ [mmol/L]", "CCPP": "CCPP [mmol/L]",
             "Ionic strength": "Ionic strength [mmol/L]",
             "Temperature": "Temperature [°C]"}[out4]
    st.plotly_chart(stage_figure(names, [getter(z) for z in seq], ylab4,
                                 si=(out4 == "SI")), key="t4")
    if out4 == "SI":
        st.caption("Marker colour shows the condition at each stage: teal = aggressive, "
                   "green = at equilibrium, orange = scaling.")

with t5:
    st.markdown(
        f'<div class="ionrow">{svg_carbonate(52)}'
        f'<div class="ioncap"><b>CO₃²⁻ — trigonal planar, D₃ₕ.</b> Three identical C–O bonds '
        f'of 1.28 Å at exactly 120°, drawn as the delocalised hybrid: bond order 1⅓ on every '
        f'bond and −⅔ charge on every oxygen, not one double bond and two singles.</div></div>',
        unsafe_allow_html=True)
    st.plotly_chart(
        bjerrum_figure(s["T"], s["I"], s["pH"], [s["a0"], s["a1"], s["a2"]]),
        key="t5")
    st.caption(
        f"Distribution of total inorganic carbon across pH, at {s['T']:.0f} °C and the "
        f"ionic strength of this water ({s['I']*1000:.2f} mmol/L). Markers show "
        f"{st.session_state.unit}. Ionic strength shifts the crossover points — "
        "that shift is exactly what the activity correction accounts for."
    )

with t6:
    st.markdown(
        '<p class="note">Consumer SI across the two dominant dose axes, with the '
        'equilibrium contour drawn. Everything else stays at the current settings.</p>',
        unsafe_allow_html=True)
    _mapped = st.session_state.get("map_on")
    if st.button("Recompute map" if _mapped else "Compute operating map",
                 key="map_btn", type="primary"):
        st.session_state["map_on"] = True
    if st.session_state.get("map_on"):
        n = 17
        lime_ax = np.linspace(0, DOSE_MAX["CaOH2"], n)
        co2_ax = np.linspace(0, DOSE_MAX["CO2"], n)
        with st.spinner("Mapping the operating plane…"):
            Z = np.empty((n, n))
            for i, c in enumerate(co2_ax):
                for j, l in enumerate(lime_ax):
                    st_ = consumer_only(**base_inputs,
                                        doses={"CaOH2": float(l), "CO2": float(c),
                                               "CaCO3": d_calc, "MgCl2": d_mg},
                                        fast=True)
                    Z[i, j] = np.nan if st_ is None else st_["SI"]
        lim = float(np.nanmax(np.abs(Z))) or 1.0
        fig = go.Figure(go.Heatmap(
            x=lime_ax, y=co2_ax, z=Z, colorscale=SI_COLORSCALE,
            zmid=0, zmin=-lim, zmax=lim, zsmooth="best",
            colorbar=dict(title=dict(text="SI", font=dict(size=11, color=INK_MUTED)),
                          tickfont=dict(size=10, color=INK_MUTED), outlinewidth=0,
                          orientation="h", y=1.045, x=0, len=0.36, thickness=10,
                          xanchor="left", yanchor="bottom"),
            hovertemplate="Ca(OH)₂ %{x:.0f} · CO₂ %{y:.0f} mg/L<br>SI %{z:+.2f}<extra></extra>"))
        fig.add_trace(go.Contour(
            x=lime_ax, y=co2_ax, z=Z, showscale=False, contours=dict(
                start=0, end=0, size=1, coloring="none",
                showlabels=True, labelfont=dict(size=11, color=INK)),
            line=dict(color=INK, width=2), hoverinfo="skip", name="SI = 0"))
        fig.add_trace(go.Scatter(
            x=[d_lime], y=[d_co2], mode="markers", showlegend=False,
            marker=dict(size=14, color="white", line=dict(color=INK, width=2.5),
                        symbol="circle"),
            hovertemplate="current setting<extra></extra>"))
        style(fig, "Ca(OH)₂ dose [mg/L]", "CO₂ dose [mg/L]", 560)
        # both axes carry the same unit over the same range, so they must be
        # scaled equally or the SI = 0 contour misstates the dose trade-off
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.add_annotation(x=d_lime, y=d_co2, text="current doses", ax=48, ay=-38,
                           showarrow=True, arrowwidth=1.4, arrowhead=0,
                           arrowcolor=INK, font=dict(size=11, color=INK))
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, key="t6")
        st.caption("The dark contour is calcite equilibrium at the consumer — every "
                   "point on it delivers stable water. The ring marks the current doses.")

with st.expander("Full calculated state"):
    st.dataframe(pd.DataFrame([{
        "Stage": z["name"], "pH": z["pH"],
        "Alk [mg/L CaCO₃]": z["Alk_mg"], "Ca [mg/L]": z["Ca_mg"],
        "Mg [mg/L]": z["Mg_mg"], "TH [mg/L CaCO₃]": z["TH"],
        "Cₜ [mol/L]": z["CT"], "SI": z["SI"], "CCPP [mol/L]": z.get("CCPP", np.nan),
        "I [mol/L]": z["I"], "γ₂": z["g2"], "T [°C]": z["T"],
    } for z in (initial, remin, res, pipeout)]), hide_index=True)
