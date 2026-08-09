
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Water Chemistry Simulator",
    page_icon="💧",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {min-width: 335px; max-width: 335px;}
.hero {
    padding: 1rem 1.25rem;
    border-radius: 18px;
    background: linear-gradient(135deg,#f3fbff,#e9f6fa);
    border: 1px solid #d6e7ee;
    margin-bottom: 1rem;
}
.hero h1 {font-size: 2rem; margin: 0 0 .25rem 0;}
.hero p {margin:0; opacity:.78;}
.flow-card {
    border:1px solid #d9e7ec;
    border-radius:18px;
    background:#fbfdfe;
    padding:.55rem .55rem .25rem .55rem;
    min-height:180px;
    box-shadow:0 3px 12px rgba(25,65,80,.05);
}
.flow-name {text-align:center;font-weight:700;margin-top:.2rem;font-size:.92rem;}
.statusbox {
    border:1px solid #d9e6eb;
    border-radius:15px;
    padding:.8rem 1rem;
    background:#fbfdfe;
    margin:.5rem 0 1rem 0;
}
.small {font-size:.88rem;opacity:.78;}
div[data-testid="stMetric"] {
    border:1px solid #e0eaee;
    border-radius:14px;
    padding:.55rem .7rem;
    background:white;
}
div[data-testid="stMetricValue"] {
    font-size: 1.20rem !important;
    line-height: 1.25 !important;
}
div[data-testid="stMetricValue"] > div {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS / APPROVED MODEL ASSUMPTIONS
# ============================================================
R = 8.314462618
R_GAS = 8.205736e-5       # m3 atm mol-1 K-1
RHO_W = 1000.0
CP_W = 4180.0

MW = {
    "CaOH2": 74.09268,
    "CO2": 44.0095,
    "CaCO3": 100.0869,
    "MgCl2": 95.211,
    "Ca": 40.078,
    "Mg": 24.305,
}

K1_25 = 10**(-6.35)
K2_25 = 10**(-10.33)
KW_25 = 1e-14
KSP_25 = 10**(-8.48)
KH_25 = 0.0339            # mol/L/atm, C = KH*p

# Van't Hoff enthalpy parameters [J/mol]
DH_K1 = 7.7e3
DH_K2 = 14.9e3
DH_KW = 55.8e3
DH_KSP = -9.4e3
DH_KH = -19.4e3

# Model constants
KLA = 1.0e-4              # s^-1
KS_SCALE = 1.0e-8         # mol L^-1 s^-1 per unit positive SI
U_EXPOSED = 8.0           # W m^-2 K^-1
U_INSULATED = 1.0         # W m^-2 K^-1
PCO2_GAS_0 = 420e-6       # atm

CHEM_DISPLAY_TO_KEY = {
    "Ca(OH)₂": "CaOH2",
    "CO₂": "CO2",
    "CaCO₃": "CaCO3",
    "MgCl₂": "MgCl2",
}
CHEM_KEY_TO_DISPLAY = {v: k for k, v in CHEM_DISPLAY_TO_KEY.items()}

# ============================================================
# CHEMISTRY
# ============================================================
def vanthoff(Kref, dH, T_C):
    T = T_C + 273.15
    Tref = 298.15
    return Kref * math.exp((-dH/R) * (1/T - 1/Tref))

def constants(T_C):
    return {
        "K1": vanthoff(K1_25, DH_K1, T_C),
        "K2": vanthoff(K2_25, DH_K2, T_C),
        "Kw": vanthoff(KW_25, DH_KW, T_C),
        "Ksp": vanthoff(KSP_25, DH_KSP, T_C),
        "KH": vanthoff(KH_25, DH_KH, T_C),
    }

def alk_mg_to_eq(x):
    return x / 50000.0

def alk_eq_to_mg(x):
    return x * 50000.0

def mg_to_mol(x, mw):
    return x / (mw * 1000.0)

def mol_to_mg(x, mw):
    return x * mw * 1000.0

def dose_to_mol(dose, mw):
    return dose / (mw * 1000.0)

def carbonate(pH, CT, T_C):
    K = constants(T_C)
    H = 10**(-pH)
    den = H**2 + K["K1"]*H + K["K1"]*K["K2"]
    a0 = H**2/den
    a1 = K["K1"]*H/den
    a2 = K["K1"]*K["K2"]/den
    return {
        "H": H,
        "OH": K["Kw"]/H,
        "a0": a0, "a1": a1, "a2": a2,
        "CO2": a0*CT,
        "HCO3": a1*CT,
        "CO3": a2*CT,
    }

def alk_calc(pH, CT, T_C):
    s = carbonate(pH, CT, T_C)
    return s["HCO3"] + 2*s["CO3"] + s["OH"] - s["H"]

def solve_pH(Alk, CT, T_C):
    if CT <= 0:
        return np.nan
    def f(pH):
        return alk_calc(pH, CT, T_C) - Alk

    xs = np.linspace(2, 14, 121)
    fs = [f(x) for x in xs]
    for i in range(len(xs)-1):
        if fs[i] == 0:
            return float(xs[i])
        if fs[i]*fs[i+1] < 0:
            return float(brentq(f, xs[i], xs[i+1], xtol=1e-10))
    return np.nan

def calc_si(Ca, CO3, T_C):
    if Ca <= 0 or CO3 <= 0:
        return np.nan
    return math.log10((Ca*CO3)/constants(T_C)["Ksp"])

def hardness(Ca_mg, Mg_mg):
    return 2.497*Ca_mg + 4.118*Mg_mg

def state_basic(name, Ca, Mg, CT, Alk, T_C, pH_override=None):
    pH = pH_override if pH_override is not None else solve_pH(Alk, CT, T_C)
    if np.isnan(pH):
        sp = {k: np.nan for k in ["CO2","HCO3","CO3","a0","a1","a2"]}
        SI = np.nan
    else:
        sp = carbonate(pH, CT, T_C)
        SI = calc_si(Ca, sp["CO3"], T_C)

    Ca_mg = mol_to_mg(Ca, MW["Ca"])
    Mg_mg = mol_to_mg(Mg, MW["Mg"])
    return {
        "name": name,
        "pH": pH,
        "Alk": Alk,
        "Alk_mg": alk_eq_to_mg(Alk),
        "Ca": Ca,
        "Ca_mg": Ca_mg,
        "Mg": Mg,
        "Mg_mg": Mg_mg,
        "TH": hardness(Ca_mg, Mg_mg),
        "CT": CT,
        "T": T_C,
        "SI": SI,
        "CO2": sp["CO2"],
        "HCO3": sp["HCO3"],
        "CO3": sp["CO3"],
        "a0": sp["a0"],
        "a1": sp["a1"],
        "a2": sp["a2"],
    }

def solve_ccpp(s):
    """Find y such that SI=0 with Ca-y, CT-y, Alk-2y."""
    if np.isnan(s["SI"]):
        return np.nan
    if abs(s["SI"]) < 1e-8:
        return 0.0

    Ca0, CT0, Alk0, T = s["Ca"], s["CT"], s["Alk"], s["T"]

    def g(y):
        Ca = Ca0 - y
        CT = CT0 - y
        Alk = Alk0 - 2*y
        if Ca <= 0 or CT <= 0:
            return np.nan
        ph = solve_pH(Alk, CT, T)
        if np.isnan(ph):
            return np.nan
        co3 = carbonate(ph, CT, T)["CO3"]
        return calc_si(Ca, co3, T)

    # precipitation
    if s["SI"] > 0:
        hi = min(Ca0, CT0) * 0.999
        if Alk0 > 0:
            hi = min(hi, Alk0/2*0.999)
        if hi <= 0:
            return np.nan
        ys = np.linspace(0, hi, 80)
    else:
        # dissolution search range
        ys = np.linspace(-0.02, 0, 100)

    prev_y, prev_f = None, None
    for y in ys:
        fy = g(y)
        if np.isnan(fy):
            continue
        if abs(fy) < 1e-8:
            return float(y)
        if prev_f is not None and prev_f*fy < 0:
            return float(brentq(g, prev_y, y))
        prev_y, prev_f = y, fy
    return np.nan

def with_ccpp(s):
    out = dict(s)
    out["CCPP"] = solve_ccpp(s)
    return out

# ============================================================
# PROCESS
# ============================================================
def remineralize(initial, doses, compute_ccpp=True):
    n_lime = dose_to_mol(doses["CaOH2"], MW["CaOH2"])
    n_co2 = dose_to_mol(doses["CO2"], MW["CO2"])
    n_calcite = dose_to_mol(doses["CaCO3"], MW["CaCO3"])
    n_mg = dose_to_mol(doses["MgCl2"], MW["MgCl2"])

    Ca = initial["Ca"] + n_lime + n_calcite
    Mg = initial["Mg"] + n_mg
    CT = initial["CT"] + n_co2 + n_calcite
    Alk = initial["Alk"] + 2*n_lime + 2*n_calcite

    s = state_basic("Remineralization outlet", Ca, Mg, CT, Alk, initial["T"])
    return with_ccpp(s) if compute_ccpp else s

def reservoir(inlet, t_h, Vw_m3, Vhs_m3, T_C, compute_ccpp=True, nsteps=60):
    Ca, Mg, CT, Alk = inlet["Ca"], inlet["Mg"], inlet["CT"], inlet["Alk"]
    T_K = T_C + 273.15
    Vw_L = Vw_m3*1000

    n_gas = PCO2_GAS_0 * Vhs_m3 / (R_GAS*T_K)
    t_s = t_h*3600

    if t_s > 0:
        nsteps = max(4, int(nsteps))
        dt = t_s/nsteps
        for _ in range(nsteps):
            s = state_basic("tmp", Ca, Mg, CT, Alk, T_C)
            if np.isnan(s["CO2"]):
                break
            pco2 = n_gas*R_GAS*T_K/max(Vhs_m3,1e-12)
            Cstar = constants(T_C)["KH"] * pco2
            dC = KLA*(s["CO2"] - Cstar)*dt

            # numerical protection
            if dC > 0:
                dC = min(dC, CT*0.20)
            else:
                dC = max(dC, -n_gas/max(Vw_L,1e-30))

            CT = max(CT-dC, 1e-15)
            n_gas = max(n_gas + dC*Vw_L, 0)

    out = state_basic("Closed reservoir outlet", Ca, Mg, CT, Alk, T_C)
    if compute_ccpp:
        out = with_ccpp(out)

    extra = {
        "pCO2": n_gas*R_GAS*T_K/max(Vhs_m3,1e-12),
        "residence_h": t_h
    }
    return out, extra

def pipe(inlet, L, D, Q_m3h, Tenv, insulated, nseg=80, compute_ccpp=True):
    Q = Q_m3h/3600
    A = math.pi*D**2/4
    v = Q/A
    ttot = L/v

    U = U_INSULATED if insulated else U_EXPOSED
    kT = 4*U/(RHO_W*CP_W*D)

    Ca, Mg, CT, Alk, T = inlet["Ca"], inlet["Mg"], inlet["CT"], inlet["Alk"], inlet["T"]
    dt = ttot/nseg
    dx = L/nseg
    cum = 0.0
    rows = []

    for i in range(nseg+1):
        x = i*dx
        s = state_basic("pipe", Ca, Mg, CT, Alk, T)
        rows.append({
            "x":x, "pH":s["pH"], "SI":s["SI"], "T":T,
            "cum_precip":cum
        })
        if i == nseg:
            break

        T = Tenv + (T-Tenv)*math.exp(-kT*dt)
        s2 = state_basic("pipe", Ca, Mg, CT, Alk, T)
        r = KS_SCALE*max(s2["SI"],0) if not np.isnan(s2["SI"]) else 0
        dy = r*dt
        dy = min(dy, Ca*0.999, CT*0.999)
        if Alk > 0:
            dy = min(dy, Alk/2*0.999)
        dy = max(dy,0)

        Ca -= dy
        CT -= dy
        Alk -= 2*dy
        cum += dy

    out = state_basic("Pipe outlet / Consumer", Ca, Mg, CT, Alk, T)
    if compute_ccpp:
        out = with_ccpp(out)

    extra = {"A":A, "v":v, "t_s":ttot, "U":U, "precip":cum}
    return out, pd.DataFrame(rows), extra

# ============================================================
# SVG PROCESS UNIT ILLUSTRATIONS
# ============================================================
def svg_desal():
    return """<svg width="100%" height="115" viewBox="0 0 220 115">
    <rect x="22" y="62" width="176" height="33" rx="7" fill="#dff3f8" stroke="#5f899a" stroke-width="2"/>
    <rect x="34" y="25" width="50" height="58" rx="6" fill="#f8fbfc" stroke="#5f899a" stroke-width="2"/>
    <line x1="47" y1="34" x2="47" y2="72" stroke="#4a8197" stroke-width="4"/>
    <line x1="59" y1="34" x2="59" y2="72" stroke="#4a8197" stroke-width="4"/>
    <line x1="71" y1="34" x2="71" y2="72" stroke="#4a8197" stroke-width="4"/>
    <path d="M85 53 H130 V39 H165 V63 H201" fill="none" stroke="#397a92" stroke-width="5"/>
    <circle cx="166" cy="40" r="12" fill="#fff" stroke="#397a92" stroke-width="2"/>
    </svg>"""

def svg_remin():
    return """<svg width="100%" height="115" viewBox="0 0 220 115">
    <rect x="55" y="26" width="110" height="75" rx="14" fill="#edf8fb" stroke="#5f899a" stroke-width="2"/>
    <ellipse cx="110" cy="27" rx="55" ry="12" fill="#fbfdfe" stroke="#5f899a" stroke-width="2"/>
    <path d="M57 69 Q82 58 110 69 T163 69 V99 H57Z" fill="#bfe9f3"/>
    <rect x="18" y="25" width="25" height="52" rx="5" fill="#fbfdfe" stroke="#5f899a" stroke-width="2"/>
    <rect x="178" y="25" width="25" height="52" rx="5" fill="#fbfdfe" stroke="#5f899a" stroke-width="2"/>
    <path d="M31 78 V94 H80" fill="none" stroke="#397a92" stroke-width="4"/>
    <path d="M190 78 V94 H140" fill="none" stroke="#397a92" stroke-width="4"/>
    <circle cx="102" cy="53" r="5" fill="#58b8d0"/><circle cx="120" cy="58" r="4" fill="#58b8d0"/>
    </svg>"""

def svg_res():
    return """<svg width="100%" height="115" viewBox="0 0 220 115">
    <rect x="51" y="18" width="118" height="89" rx="16" fill="#fbfdfe" stroke="#587f90" stroke-width="2.5"/>
    <ellipse cx="110" cy="20" rx="59" ry="13" fill="#f5fafb" stroke="#587f90" stroke-width="2.5"/>
    <path d="M53 68 Q80 58 110 68 T167 68 V103 H53Z" fill="#bce8f2"/>
    <line x1="53" y1="57" x2="167" y2="57" stroke="#91aab4" stroke-width="1.5" stroke-dasharray="5,5"/>
    <circle cx="82" cy="45" r="3" fill="#8aabb7"/><circle cx="108" cy="39" r="3" fill="#8aabb7"/><circle cx="134" cy="47" r="3" fill="#8aabb7"/>
    <text x="82" y="53" font-size="10" fill="#667f89">HEADSPACE</text>
    </svg>"""

def svg_pipe():
    return """<svg width="100%" height="115" viewBox="0 0 220 115">
    <path d="M17 63 H78 Q94 63 94 79 V87 Q94 98 108 98 H202" fill="none" stroke="#6e8f9b" stroke-width="22"/>
    <path d="M17 63 H78 Q94 63 94 79 V87 Q94 98 108 98 H202" fill="none" stroke="#bde8f2" stroke-width="13"/>
    <path d="M34 63 h30" stroke="#388aa6" stroke-width="3"/><path d="M56 57 l9 6 l-9 6" fill="none" stroke="#388aa6" stroke-width="3"/>
    <rect x="119" y="25" width="62" height="36" rx="6" fill="#fafcfd" stroke="#6e8f9b" stroke-width="2"/>
    <path d="M129 49 q10 -18 20 0 q10 -18 20 0" fill="none" stroke="#cc8c5d" stroke-width="3"/>
    </svg>"""

def svg_consumer():
    return """<svg width="100%" height="115" viewBox="0 0 220 115">
    <path d="M58 62 L110 22 L162 62 V105 H58Z" fill="#f8fcfd" stroke="#5f899a" stroke-width="2.5"/>
    <rect x="94" y="72" width="32" height="33" fill="#d6edf3" stroke="#5f899a" stroke-width="2"/>
    <rect x="69" y="50" width="20" height="18" fill="#d6edf3" stroke="#5f899a" stroke-width="2"/>
    <rect x="136" y="50" width="17" height="18" fill="#d6edf3" stroke="#5f899a" stroke-width="2"/>
    <path d="M15 88 H58" stroke="#3b8ca8" stroke-width="6"/><path d="M49 80 l10 8 l-10 8" fill="none" stroke="#3b8ca8" stroke-width="3"/>
    </svg>"""

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
<h1>💧 Interactive Remineralization & Water Transport Simulator</h1>
<p>
Enter the initial water quality and operating conditions in the left sidebar, then adjust the
remineralization doses, reservoir conditions and pipe parameters. Click any process-unit icon
to inspect the calculated water quality at that stage, and use the analysis tabs below to explore
how the selected inputs affect the process outputs.
</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# AUTO-STABILIZATION
# ============================================================

STABLE_SI_TOL = 0.02
STABLE_PH_MIN = 7.0
STABLE_PH_MAX = 8.5
AUTO_MAX_EVAL_FAST = 120
AUTO_MAX_EVAL_REFINE = 60


class StablePointFound(Exception):
    """Internal early-stop signal used by the optimizer."""
    def __init__(self, x, state):
        self.x = np.array(x, dtype=float)
        self.state = state


def evaluate_consumer_for_doses(
    initial_state,
    test_doses,
    t_res_h,
    Vw_m3,
    Vhs_m3,
    Tres_C,
    L_m,
    D_m,
    Q_m3h,
    Tenv_C,
    insulated_pipe,
    fast=True,
):
    """
    Run the same approved process model for a candidate dose set.
    During optimization a lower spatial/time resolution is used only to
    reduce computation time; the governing equations are unchanged.
    """
    rr = remineralize(initial_state, test_doses, compute_ccpp=False)

    res_steps = 14 if fast else 60
    pipe_segments = 10 if fast else 80

    rres, _ = reservoir(
        rr,
        t_res_h,
        Vw_m3,
        Vhs_m3,
        Tres_C,
        compute_ccpp=False,
        nsteps=res_steps,
    )

    pout, _, _ = pipe(
        rres,
        L_m,
        D_m,
        Q_m3h,
        Tenv_C,
        insulated_pipe,
        nseg=pipe_segments,
        compute_ccpp=False,
    )
    return pout


def is_stable_state(state):
    if state is None:
        return False
    if np.isnan(state["SI"]) or np.isnan(state["pH"]):
        return False
    return (
        abs(state["SI"]) <= STABLE_SI_TOL
        and STABLE_PH_MIN <= state["pH"] <= STABLE_PH_MAX
    )


def stabilization_objective(
    x,
    initial_state,
    current_doses,
    process_args,
    fast=True,
    early_stop=True,
):
    """
    Objective:
      1) drive SI toward 0,
      2) penalize pH outside 7.0-8.5,
      3) weakly prefer smaller changes from the user's present doses.
    """
    lime, co2, calcite = [float(v) for v in x]

    test_doses = {
        "CaOH2": lime,
        "CO2": co2,
        "CaCO3": calcite,
        "MgCl2": float(current_doses["MgCl2"]),
    }

    state = evaluate_consumer_for_doses(
        initial_state,
        test_doses,
        *process_args,
        fast=fast,
    )

    if np.isnan(state["SI"]) or np.isnan(state["pH"]):
        return 1e6

    if early_stop and is_stable_state(state):
        raise StablePointFound(x, state)

    # Main target: SI = 0
    score = (state["SI"] / 0.10) ** 2

    # pH is constrained softly to the selected acceptable interval.
    if state["pH"] < STABLE_PH_MIN:
        score += 12.0 * (STABLE_PH_MIN - state["pH"]) ** 2
    elif state["pH"] > STABLE_PH_MAX:
        score += 12.0 * (state["pH"] - STABLE_PH_MAX) ** 2

    # Weak regularization: among similarly stable solutions, prefer a
    # smaller change from the user's current operating point.
    current = np.array([
        current_doses["CaOH2"],
        current_doses["CO2"],
        current_doses["CaCO3"],
    ], dtype=float)

    ranges = np.array([200.0, 200.0, 300.0], dtype=float)
    score += 0.015 * np.sum(((np.array(x) - current) / ranges) ** 2)

    return float(score)


def run_powell_stage(
    x0,
    initial_state,
    current_doses,
    process_args,
    fast,
    maxfev,
):
    bounds = [(0.0, 200.0), (0.0, 200.0), (0.0, 300.0)]

    try:
        result = minimize(
            stabilization_objective,
            x0=np.array(x0, dtype=float),
            args=(initial_state, current_doses, process_args, fast, True),
            method="Powell",
            bounds=bounds,
            options={
                "maxfev": int(maxfev),
                "maxiter": 50,
                "xtol": 0.5,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        candidate_x = np.array(result.x, dtype=float)
        candidate_state = evaluate_consumer_for_doses(
            initial_state,
            {
                "CaOH2": candidate_x[0],
                "CO2": candidate_x[1],
                "CaCO3": candidate_x[2],
                "MgCl2": current_doses["MgCl2"],
            },
            *process_args,
            fast=fast,
        )
        return candidate_x, candidate_state, result.nfev, bool(result.success)

    except StablePointFound as found:
        return found.x, found.state, None, True


def state_quality_score(state):
    if state is None or np.isnan(state["SI"]) or np.isnan(state["pH"]):
        return np.inf

    score = abs(state["SI"])
    if state["pH"] < STABLE_PH_MIN:
        score += 0.5 * (STABLE_PH_MIN - state["pH"])
    elif state["pH"] > STABLE_PH_MAX:
        score += 0.5 * (state["pH"] - STABLE_PH_MAX)
    return float(score)


def auto_stabilize(
    initial_state,
    current_doses,
    t_res_h,
    Vw_m3,
    Vhs_m3,
    Tres_C,
    L_m,
    D_m,
    Q_m3h,
    Tenv_C,
    insulated_pipe,
):
    """
    Two-stage optimization with hard computational limits.

    Stage 1:
      Fast Powell optimization using reduced reservoir/pipe resolution.

    Stage 2:
      Validate with the full-resolution model. If necessary, perform one
      bounded refinement with a limited number of evaluations.

    The function always terminates because both Powell stages have maxfev.
    """
    process_args = (
        t_res_h,
        Vw_m3,
        Vhs_m3,
        Tres_C,
        L_m,
        D_m,
        Q_m3h,
        Tenv_C,
        insulated_pipe,
    )

    x0 = np.array([
        current_doses["CaOH2"],
        current_doses["CO2"],
        current_doses["CaCO3"],
    ], dtype=float)

    # Evaluate current operating point first.
    current_full = evaluate_consumer_for_doses(
        initial_state,
        current_doses,
        *process_args,
        fast=False,
    )

    if is_stable_state(current_full):
        return {
            "doses": dict(current_doses),
            "state": current_full,
            "stable": True,
            "message": "The current operating point is already stable.",
        }

    # -------- Stage 1: fast search --------
    x_fast, fast_state, _, _ = run_powell_stage(
        x0,
        initial_state,
        current_doses,
        process_args,
        fast=True,
        maxfev=AUTO_MAX_EVAL_FAST,
    )

    fast_doses = {
        "CaOH2": float(np.clip(x_fast[0], 0, 200)),
        "CO2": float(np.clip(x_fast[1], 0, 200)),
        "CaCO3": float(np.clip(x_fast[2], 0, 300)),
        "MgCl2": float(current_doses["MgCl2"]),
    }

    # Full-resolution validation.
    full_state = evaluate_consumer_for_doses(
        initial_state,
        fast_doses,
        *process_args,
        fast=False,
    )

    if is_stable_state(full_state):
        return {
            "doses": fast_doses,
            "state": full_state,
            "stable": True,
            "message": "A stable operating point was found and validated with the full model.",
        }

    # -------- Stage 2: limited refinement --------
    # This stage starts from the fast solution but uses an intermediate
    # resolution. It also has a strict maximum number of evaluations.
    try:
        x_ref, ref_state, _, _ = run_powell_stage(
            x_fast,
            initial_state,
            current_doses,
            process_args,
            fast=False,
            maxfev=AUTO_MAX_EVAL_REFINE,
        )
    except Exception:
        x_ref = x_fast
        ref_state = full_state

    refined_doses = {
        "CaOH2": float(np.clip(x_ref[0], 0, 200)),
        "CO2": float(np.clip(x_ref[1], 0, 200)),
        "CaCO3": float(np.clip(x_ref[2], 0, 300)),
        "MgCl2": float(current_doses["MgCl2"]),
    }

    final_state = evaluate_consumer_for_doses(
        initial_state,
        refined_doses,
        *process_args,
        fast=False,
    )

    # Compare refined solution against the fast candidate and current point.
    candidates = [
        (current_doses, current_full),
        (fast_doses, full_state),
        (refined_doses, final_state),
    ]
    best_doses, best_state = min(
        candidates,
        key=lambda pair: state_quality_score(pair[1]),
    )

    return {
        "doses": dict(best_doses),
        "state": best_state,
        "stable": is_stable_state(best_state),
        "message": (
            "A stable operating point was found."
            if is_stable_state(best_state)
            else "No fully stable point was found within the evaluation limits; the best available point is shown."
        ),
    }


# Apply optimized doses BEFORE the sidebar widgets are instantiated.
# This avoids Streamlit's restriction on changing a widget after creation.
if "pending_auto_doses" in st.session_state:
    pending = st.session_state.pop("pending_auto_doses")
    st.session_state["dose_lime"] = float(pending["CaOH2"])
    st.session_state["dose_co2"] = float(pending["CO2"])
    st.session_state["dose_calcite"] = float(pending["CaCO3"])

# ============================================================
# SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Initial desalinated water")
    pH0 = st.number_input("Initial pH", 2.0, 14.0, 6.50, .05)
    alk0 = st.number_input("Initial alkalinity [mg/L as CaCO₃]", 0.0, 500.0, 20.0, 1.0)
    Ca0 = st.number_input("Initial Ca [mg/L]", 0.0, 500.0, 5.0, 1.0)
    Mg0 = st.number_input("Initial Mg [mg/L]", 0.0, 500.0, 1.0, .5)
    CT0 = st.number_input("Initial Cₜ [mol/L]", 0.000001, 0.050000, 0.000500, 0.000050, format="%.6f")
    T0 = st.number_input("Initial water temperature [°C]", 1.0, 60.0, 25.0, 1.0)

    st.divider()
    st.header("Remineralization")
    d_lime = st.slider("Ca(OH)₂ [mg/L]", 0.0, 200.0, 20.0, 1.0, key="dose_lime")
    d_co2 = st.slider("CO₂ [mg/L]", 0.0, 200.0, 15.0, 1.0, key="dose_co2")
    d_calc = st.slider("CaCO₃ [mg/L]", 0.0, 300.0, 30.0, 1.0, key="dose_calcite")
    d_mg = st.slider("MgCl₂ [mg/L]", 0.0, 200.0, 5.0, 1.0, key="dose_mgcl2")

    st.divider()
    st.header("Closed reservoir")
    t_res = st.number_input("Residence time [h]", 0.0, 72.0, 2.0, .25)
    Vw = st.number_input("Water volume [m³]", 0.01, 1e7, 1000.0, 10.0)
    Vhs = st.number_input("Headspace volume [m³]", 0.01, 1e7, 100.0, 10.0)
    Tres = st.number_input("Reservoir temperature [°C]", 1.0, 60.0, 25.0, 1.0)
    
    st.divider()
    st.header("Supply pipe")
    L = st.number_input("Length [m]", 1.0, 1e7, 5000.0, 100.0)
    D = st.number_input("Diameter [m]", 0.01, 10.0, .50, .05)
    Q = st.number_input("Flow rate [m³/h]", .01, 1e7, 500.0, 10.0)
    Tenv = st.number_input("Ambient temperature [°C]", -10.0, 60.0, 30.0, 1.0)
    insulated = st.toggle("Insulated pipe", False)

    with st.expander("Model assumptions"):
        st.write(f"kLa = {KLA:.2e} s⁻¹")
        st.write(f"kₛ = {KS_SCALE:.2e} mol/(L·s)")
        st.write(f"U exposed = {U_EXPOSED:.1f} W/(m²·K)")
        st.write(f"U insulated = {U_INSULATED:.1f} W/(m²·K)")
        st.write(f"Initial headspace pCO₂ = {PCO2_GAS_0:.2e} atm")
        st.caption("K₁, K₂, Kᵥ, Ksp and KH are temperature-corrected with Van't Hoff.")

# ============================================================
# RUN MAIN PROCESS
# ============================================================
initial = state_basic(
    "Initial desalinated water",
    mg_to_mol(Ca0, MW["Ca"]),
    mg_to_mol(Mg0, MW["Mg"]),
    CT0,
    alk_mg_to_eq(alk0),
    T0,
    pH_override=pH0
)
initial = with_ccpp(initial)

doses = {"CaOH2":d_lime, "CO2":d_co2, "CaCO3":d_calc, "MgCl2":d_mg}
remin = remineralize(initial, doses, compute_ccpp=True)
res, res_extra = reservoir(remin, t_res, Vw, Vhs, Tres, compute_ccpp=True)
pipeout, profile, pipe_extra = pipe(res, L, D, Q, Tenv, insulated, compute_ccpp=True)

# Automatic decision-support action:
# Find remineralization doses that stabilize the FINAL consumer water.
st.markdown("### Automatic stabilization")
auto_col1, auto_col2 = st.columns([1, 2])

with auto_col1:
    auto_clicked = st.button(
        "✨ Auto-stabilize water",
        type="primary",
        use_container_width=True,
        help="Automatically adjusts the remineralization doses so the final consumer water approaches SI = 0."
    )

with auto_col2:
    st.caption(
        f"The optimizer changes Ca(OH)₂, CO₂ and CaCO₃ only. "
        f"Target: |SI| ≤ {STABLE_SI_TOL:.2f} and pH between "
        f"{STABLE_PH_MIN:.1f} and {STABLE_PH_MAX:.1f}."
    )

if auto_clicked:
    current_doses_for_opt = {
        "CaOH2": float(st.session_state["dose_lime"]),
        "CO2": float(st.session_state["dose_co2"]),
        "CaCO3": float(st.session_state["dose_calcite"]),
        "MgCl2": float(st.session_state["dose_mgcl2"]),
    }

    with st.spinner("Searching for a stable operating point..."):
        result = auto_stabilize(
            initial,
            current_doses_for_opt,
            t_res, Vw, Vhs, Tres,
            L, D, Q, Tenv, insulated
        )

    opt = result["doses"]
    final_state = result["state"]

    # Store new widget values for the NEXT run, then rerun.
    st.session_state["pending_auto_doses"] = {
        "CaOH2": round(opt["CaOH2"], 1),
        "CO2": round(opt["CO2"], 1),
        "CaCO3": round(opt["CaCO3"], 1),
    }

    status_word = "Stable solution" if result["stable"] else "Best available solution"
    st.session_state["auto_message"] = (
        f'{status_word}: Ca(OH)₂ = {opt["CaOH2"]:.1f} mg/L, '
        f'CO₂ = {opt["CO2"]:.1f} mg/L, '
        f'CaCO₃ = {opt["CaCO3"]:.1f} mg/L. '
        f'Final consumer SI = {final_state["SI"]:.2f}, '
        f'pH = {final_state["pH"]:.2f}. '
        f'{result["message"]}'
    )
    st.session_state["auto_message_stable"] = bool(result["stable"])

    # Auto-stabilization targets FINAL consumer water.
    # Therefore, after the optimization, show the Consumer state automatically.
    st.session_state["unit"] = "Consumer"
    st.query_params["unit"] = "consumer"
    st.rerun()

if "auto_message" in st.session_state:
    if st.session_state.get("auto_message_stable", False):
        st.success(st.session_state["auto_message"])
    else:
        st.warning(st.session_state["auto_message"])

# ============================================================
# PROCESS FLOW ILLUSTRATION
# ============================================================
st.subheader("Interactive process flow")

if "unit" not in st.session_state:
    st.session_state.unit = "Remineralization"

# Clicking directly on each equipment illustration selects that unit.
unit_from_url = st.query_params.get("unit", None)
url_to_unit = {
    "initial": "Initial",
    "remineralization": "Remineralization",
    "reservoir": "Closed reservoir",
    "pipe": "Supply pipe",
    "consumer": "Consumer",
}
if unit_from_url in url_to_unit:
    st.session_state.unit = url_to_unit[unit_from_url]

units = [
    ("Desalinated water", svg_desal(), "Initial", "initial"),
    ("Remineralization", svg_remin(), "Remineralization", "remineralization"),
    ("Closed reservoir", svg_res(), "Closed reservoir", "reservoir"),
    ("Supply pipe", svg_pipe(), "Supply pipe", "pipe"),
    ("Consumer", svg_consumer(), "Consumer", "consumer"),
]

cols = st.columns(5)
for col, (label, svg, key, slug) in zip(cols, units):
    with col:
        st.markdown(
            f"""
            <a href="?unit={slug}" target="_self"
               style="display:block; text-decoration:none; cursor:pointer;">
                {svg}
            </a>
            <div class="flow-name">{label}</div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# OUTPUTS
# ============================================================
states = {
    "Initial": initial,
    "Remineralization": remin,
    "Closed reservoir": res,
    "Supply pipe": pipeout,
    "Consumer": pipeout,
}
s = states[st.session_state.unit]

st.divider()
st.subheader(f"Calculated water quality — {st.session_state.unit}")

if np.isnan(s["SI"]):
    condition = "⚪ Not available"
elif abs(s["SI"]) <= STABLE_SI_TOL:
    condition = "🟢 Stable / near calcite equilibrium"
elif s["SI"] > STABLE_SI_TOL:
    condition = "🔴 Oversaturated — scaling tendency"
else:
    condition = "🟠 Undersaturated — dissolution/aggressive tendency"

st.markdown(f'<div class="statusbox"><b>Water condition:</b> {condition}</div>', unsafe_allow_html=True)

# Quantitative, model-based recommendations.
# Each recommendation changes ONE sidebar remineralization dose at a time,
# while holding all other current inputs fixed, and searches for a setting
# that moves SI of the currently selected process unit closer to zero.

def selected_state_for_doses(test_doses):
    """Recalculate the currently inspected process unit for a candidate dose set."""
    rr = remineralize(initial, test_doses, compute_ccpp=False)

    if st.session_state.unit == "Remineralization":
        return rr

    rres, _ = reservoir(rr, t_res, Vw, Vhs, Tres, compute_ccpp=False)

    if st.session_state.unit == "Closed reservoir":
        return rres

    if st.session_state.unit in ["Supply pipe", "Consumer"]:
        pout, _, _ = pipe(
            rres, L, D, Q, Tenv, insulated,
            nseg=30, compute_ccpp=False
        )
        return pout

    return initial


def stability_score(state):
    """Lower score = closer to calcite equilibrium; mild pH penalty outside 7–8.5."""
    if np.isnan(state["SI"]):
        return np.inf

    score = abs(state["SI"])

    if not np.isnan(state["pH"]):
        if state["pH"] < 7.0:
            score += 0.35 * (7.0 - state["pH"])
        elif state["pH"] > 8.5:
            score += 0.35 * (state["pH"] - 8.5)

    return score


dose_specs = {
    "CaOH2": {
        "label": "Ca(OH)₂",
        "current": d_lime,
        "min": 0.0,
        "max": 200.0,
    },
    "CO2": {
        "label": "CO₂",
        "current": d_co2,
        "min": 0.0,
        "max": 200.0,
    },
    "CaCO3": {
        "label": "CaCO₃",
        "current": d_calc,
        "min": 0.0,
        "max": 300.0,
    },
    "MgCl2": {
        "label": "MgCl₂",
        "current": d_mg,
        "min": 0.0,
        "max": 200.0,
    },
}

current_score = stability_score(s)
candidate_recommendations = []

# MgCl2 affects hardness but is not used directly in the calcite SI equation,
# so the automatic SI-targeting recommendations scan the three carbonate-system chemicals.
for chem in ["CaOH2", "CO2", "CaCO3"]:
    spec = dose_specs[chem]
    values = np.linspace(spec["min"], spec["max"], 31)

    best_value = spec["current"]
    best_state = s
    best_score = current_score

    for value in values:
        td = dict(doses)
        td[chem] = float(value)
        cand_state = selected_state_for_doses(td)
        cand_score = stability_score(cand_state)

        if cand_score < best_score:
            best_score = cand_score
            best_value = float(value)
            best_state = cand_state

    improvement = current_score - best_score

    # Only recommend a genuine change that improves the current model state.
    if improvement > 1e-4 and abs(best_value - spec["current"]) > 0.5:
        candidate_recommendations.append({
            "chem": chem,
            "label": spec["label"],
            "value": best_value,
            "si": best_state["SI"],
            "pH": best_state["pH"],
            "improvement": improvement,
        })

candidate_recommendations.sort(key=lambda x: x["improvement"], reverse=True)

st.markdown(f"#### Recommended adjustments — {st.session_state.unit}")

selected_is_stable = (
    not np.isnan(s["SI"])
    and not np.isnan(s["pH"])
    and abs(s["SI"]) <= STABLE_SI_TOL
    and STABLE_PH_MIN <= s["pH"] <= STABLE_PH_MAX
)

if selected_is_stable:
    st.markdown(
        f"- **No chemical adjustment is currently required for {st.session_state.unit}.** "
        f"The calculated state is within the simulator's stability target "
        f"(**|SI| ≤ {STABLE_SI_TOL:.2f}**, pH between {STABLE_PH_MIN:.1f} and {STABLE_PH_MAX:.1f})."
    )
    st.markdown(
        "- Keep the current remineralization doses and inspect the other process units "
        "to see how water chemistry changes along the system."
    )
elif candidate_recommendations:
    for rec in candidate_recommendations[:2]:
        st.markdown(
            f'- Set **{rec["label"]} to approximately {rec["value"]:.0f} mg/L** '
            f'in the Remineralization section. With all other current inputs unchanged, '
            f'the model predicts **SI ≈ {rec["si"]:.2f}** and **pH ≈ {rec["pH"]:.2f}** '
            f'at **{st.session_state.unit}**.'
        )

    if len(candidate_recommendations) == 1:
        rec = candidate_recommendations[0]
        st.markdown(
            f'- Use **{rec["label"]} = {rec["value"]:.0f} mg/L** as a starting point, then '
            f'fine-tune nearby values while monitoring the selected unit; '
            f'the target is **|SI| ≤ {STABLE_SI_TOL:.2f}**.'
        )
else:
    st.markdown(
        f"- No single-dose change produced a substantial improvement for "
        f"**{st.session_state.unit}** within the scanned ranges."
    )
    st.markdown(
        "- Try the **Auto-stabilize water** function if the goal is to stabilize the "
        "final water delivered to the Consumer."
    )

st.caption(
    "Suggested numerical settings are model-based operating points for this simulator, "
    "not regulatory or plant-design limits."
)

r1 = st.columns(5)
r1[0].metric("pH", "N/A" if np.isnan(s["pH"]) else f'{s["pH"]:.2f}')
r1[1].metric("Alkalinity", f'{s["Alk_mg"]:.2f} mg/L as CaCO₃')
r1[2].metric("Ca", f'{s["Ca_mg"]:.2f} mg/L')
r1[3].metric("Mg", f'{s["Mg_mg"]:.2f} mg/L')
r1[4].metric("Total hardness", f'{s["TH"]:.2f} mg/L as CaCO₃')

r2 = st.columns(5)
r2[0].metric("Cₜ", f'{s["CT"]*1000:.2f} mmol/L')
r2[1].metric("SI", "N/A" if np.isnan(s["SI"]) else f'{s["SI"]:.2f}')
r2[2].metric("CCPP", "N/A" if np.isnan(s["CCPP"]) else f'{s["CCPP"]*1000:.2f} mmol/L')
r2[3].metric("Dissolved CO₂*", "N/A" if np.isnan(s["CO2"]) else f'{s["CO2"]*1000:.2f} mmol/L')
r2[4].metric("Temperature", f'{s["T"]:.2f} °C')

if st.session_state.unit == "Closed reservoir":
    st.caption(f'Calculated final headspace pCO₂ = {res_extra["pCO2"]:.2e} atm')

if st.session_state.unit == "Supply pipe":
    c = st.columns(4)
    c[0].metric("Velocity", f'{pipe_extra["v"]:.2f} m/s')
    c[1].metric("Residence time", f'{pipe_extra["t_s"]/60:.2f} min')
    c[2].metric("U", f'{pipe_extra["U"]:.2f} W/(m²·K)')
    c[3].metric("CaCO₃ precipitated", f'{pipe_extra["precip"]*1000:.2f} mmol/L')

st.markdown("#### Carbonate species")
if not np.isnan(s["a0"]):
    species_names = ["CO₂*", "HCO₃⁻", "CO₃²⁻"]
    species_fractions = [s["a0"], s["a1"], s["a2"]]

    fig_species, ax_species = plt.subplots(figsize=(8, 3.6))
    ax_species.bar(species_names, species_fractions)
    ax_species.set_xlabel("Carbonate species")
    ax_species.set_ylabel("Fraction of total inorganic carbon")
    ax_species.set_title("Carbonate species distribution")
    ax_species.tick_params(axis="x", labelrotation=0)
    ax_species.set_ylim(bottom=0)
    ax_species.grid(axis="y", alpha=.25)
    st.pyplot(fig_species, use_container_width=True)

# ============================================================
# GRAPHS
# ============================================================
st.divider()
st.header("Simulation analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Remineralization",
    "🛢️ Closed reservoir",
    "〰️ Supply pipe",
    "💧 Complete system"
])

with tab1:
    a,b = st.columns(2)
    chem_display = a.selectbox(
        "Chemical to vary",
        ["Ca(OH)₂", "CO₂", "CaCO₃", "MgCl₂"]
    )
    chem = CHEM_DISPLAY_TO_KEY[chem_display]
    out = b.selectbox("Output", ["Alkalinity","pH","SI","CCPP","Total hardness"])

    xmax = {"CaOH2":200, "CO2":200, "CaCO3":300, "MgCl2":200}[chem]
    xs = np.linspace(0,xmax,45)
    ys = []
    for x in xs:
        d = dict(doses)
        d[chem] = x
        need_ccpp = out == "CCPP"
        rr = remineralize(initial,d,compute_ccpp=need_ccpp)
        if out == "Alkalinity":
            ys.append(rr["Alk_mg"]); ylabel="Alkalinity [mg/L as CaCO₃]"
        elif out == "pH":
            ys.append(rr["pH"]); ylabel="pH"
        elif out == "SI":
            ys.append(rr["SI"]); ylabel="SI"
        elif out == "CCPP":
            ys.append(rr["CCPP"]*1000 if not np.isnan(rr["CCPP"]) else np.nan); ylabel="CCPP [mmol/L]"
        else:
            ys.append(rr["TH"]); ylabel="Total hardness [mg/L as CaCO₃]"

    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(xs,ys,linewidth=2)
    ax.set_xlabel(f"{CHEM_KEY_TO_DISPLAY[chem]} dose [mg/L]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{out} as a function of {CHEM_KEY_TO_DISPLAY[chem]} dose")
    ax.grid(alpha=.25)
    if out=="SI": ax.axhline(0,linestyle="--",linewidth=1)
    st.pyplot(fig,use_container_width=True)
    st.caption("The other three chemical doses remain fixed at the current sidebar values.")

with tab2:
    out2 = st.selectbox("Output",["Dissolved CO₂","pH","Cₜ","SI","CCPP"],key="resgraph")
    ts = np.linspace(0,max(6,t_res*2),28)
    ys=[]
    for tt in ts:
        need_ccpp = out2=="CCPP"
        rr,_ = reservoir(remin,tt,Vw,Vhs,Tres,compute_ccpp=need_ccpp)
        if out2=="Dissolved CO₂":
            # Internal CO₂* concentration is mol/L; convert to mmol/L for display.
            ys.append(rr["CO2"]*1000); ylabel="CO₂* [mmol/L]"
        elif out2=="pH":
            ys.append(rr["pH"]); ylabel="pH"
        elif out2=="Cₜ":
            ys.append(rr["CT"]*1000); ylabel="Cₜ [mmol/L]"
        elif out2=="SI":
            ys.append(rr["SI"]); ylabel="SI"
        else:
            ys.append(rr["CCPP"]*1000 if not np.isnan(rr["CCPP"]) else np.nan); ylabel="CCPP [mmol/L]"
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(ts,ys,linewidth=2)
    ax.axvline(t_res,linestyle=":",linewidth=1,label="Selected residence time")
    if out2=="SI": ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_xlabel("Residence time [h]"); ax.set_ylabel(ylabel)
    ax.set_title(f"Closed reservoir: {out2} vs residence time")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig,use_container_width=True)

with tab3:
    out3=st.selectbox("Show along pipe",["Cumulative CaCO₃ precipitation","Temperature","pH","SI"])
    if out3=="Cumulative CaCO₃ precipitation":
        yy=profile["cum_precip"]*1000; yl="Cumulative CaCO₃ precipitated [mmol/L]"
    elif out3=="Temperature":
        yy=profile["T"]; yl="Temperature [C]"
    elif out3=="pH":
        yy=profile["pH"]; yl="pH"
    else:
        yy=profile["SI"]; yl="SI"
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(profile["x"],yy,linewidth=2)
    if out3=="SI": ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_xlabel("Distance along pipe [m]"); ax.set_ylabel(yl)
    ax.set_title(out3 + " along the supply pipe"); ax.grid(alpha=.25)
    st.pyplot(fig,use_container_width=True)

with tab4:
    out4=st.selectbox("Parameter",["pH","Alkalinity","Ca","Mg","Total hardness","Cₜ","SI","CCPP","Temperature"])
    ss=[initial,remin,res,pipeout]
    names=["Initial","Remineralization","Reservoir","Consumer"]
    if out4=="pH": yy=[z["pH"] for z in ss]; yl="pH"
    elif out4=="Alkalinity": yy=[z["Alk_mg"] for z in ss]; yl="Alkalinity [mg/L as CaCO₃]"
    elif out4=="Ca": yy=[z["Ca_mg"] for z in ss]; yl="Ca [mg/L]"
    elif out4=="Mg": yy=[z["Mg_mg"] for z in ss]; yl="Mg [mg/L]"
    elif out4=="Total hardness": yy=[z["TH"] for z in ss]; yl="Total hardness [mg/L as CaCO₃]"
    elif out4=="Cₜ": yy=[z["CT"]*1000 for z in ss]; yl="Cₜ [mmol/L]"
    elif out4=="SI": yy=[z["SI"] for z in ss]; yl="SI"
    elif out4=="CCPP": yy=[z["CCPP"]*1000 if not np.isnan(z["CCPP"]) else np.nan for z in ss]; yl="CCPP [mmol/L]"
    else: yy=[z["T"] for z in ss]; yl="Temperature [C]"
    fig,ax=plt.subplots(figsize=(8,4))
    ax.scatter(names,yy,s=60)
    if out4=="SI": ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_ylabel(yl); ax.set_title(f"{out4} throughout the complete system"); ax.grid(alpha=.25)
    st.pyplot(fig,use_container_width=True)

with st.expander("Full calculated state table"):
    tbl=[]
    for z in [initial,remin,res,pipeout]:
        tbl.append({
            "Stage":z["name"],"pH":z["pH"],"Alk [mg/L as CaCO₃]":z["Alk_mg"],
            "Ca [mg/L]":z["Ca_mg"],"Mg [mg/L]":z["Mg_mg"],"TH":z["TH"],
            "CT [mol/L]":z["CT"],"SI":z["SI"],"CCPP [mol/L]":z["CCPP"],"T [C]":z["T"]
        })
    st.dataframe(pd.DataFrame(tbl),hide_index=True,use_container_width=True)
