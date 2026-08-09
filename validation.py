"""
Validation suite for the Remineralization & Water Transport Simulator.
======================================================================

Run with:   python validation.py

Every check below is executed against the SHIPPED model in streamlit_app.py —
not a copy — by importing its chemistry and process functions with a stubbed
Streamlit module. Checks are grouped by what they establish:

  1  activity model            - the Davies/Debye-Huckel machinery
  2  equilibrium constants     - Van't Hoff and the conditional conversion
  3  pH solver                 - uniqueness, convergence, accuracy
  4  reduction to the original - at I=0 the new model IS the old model
  5  textbook benchmarks       - independently known answers
  6  internal consistency      - the model must not contradict itself
  7  mass balances             - nothing created or destroyed
  8  process units             - reservoir and pipe behave physically
  9  PHREEQC cross-check       - optional; skipped if phreeqpython absent

A failing check prints FAIL with the numbers involved. The exit code is the
number of failures, so this can gate a commit.
"""
import math
import sys
import types

# --- import the shipped model with a no-op Streamlit -------------------------
class _Stub:
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return self
    def __contains__(self, x): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def setdefault(self, *a): return None


def _load_model(path="streamlit_app.py", cut="# 8. PROCESS-UNIT ILLUSTRATIONS"):
    """Load only the computational half of an app file, above the UI section."""
    m = types.ModuleType("streamlit")
    m.__getattr__ = lambda n: _Stub()
    sys.modules["streamlit"] = m
    src = open(path).read().split(cut)[0]
    src = (src.replace("st.set_page_config(", "_stub(")
              .replace("st.markdown(", "_stub(")
              .replace("@st.cache_data(show_spinner=False, max_entries=512)", ""))
    ns = {"_stub": lambda *a, **k: None}
    exec(compile(src, path, "exec"), ns)
    return ns


M = _load_model()
build_state = M["build_state"]; make_initial = M["make_initial"]
remineralize = M["remineralize"]; reservoir = M["reservoir"]; pipe = M["pipe"]
solve_H = M["solve_H"]; alk_from_H = M["alk_from_H"]; speciate = M["speciate"]
conditional_constants = M["conditional_constants"]; davies_gamma = M["davies_gamma"]
dh_A = M["dh_A"]; thermo_constants = M["thermo_constants"]; solve_ccpp = M["solve_ccpp"]
CT_from_pH_alk = M["CT_from_pH_alk"]; alk_from_pH_CT = M["alk_from_pH_CT"]
MW = M["MW"]

PASS = FAIL = 0
_section = ""


def section(title):
    global _section
    _section = title
    print(f"\n{title}")
    print("-" * len(title))


def ck(label, got, want, tol, unit=""):
    """Assert |got - want| <= tol."""
    global PASS, FAIL
    good = (got == want) if tol == 0 else abs(got - want) <= tol
    if good:
        PASS += 1
        print(f"  PASS  {label:<52} {got:>12.5g} {unit}")
    else:
        FAIL += 1
        print(f"  FAIL  {label:<52} {got:>12.5g} {unit}  expected {want:.5g} +/- {tol:g}")


def ck_true(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}   {detail}")


# reference water used throughout
FEED = dict(mode="pH + alkalinity", pH0=6.50, alk0_mg=20.0, CT0=5e-4,
            Ca0_mg=5.0, Mg0_mg=1.0, nacl_mg=60.0, T0=25.0)
DOSES = {"CaOH2": 20.0, "CO2": 15.0, "CaCO3": 30.0, "MgCl2": 5.0}


def feed():
    s, _, _ = make_initial(**FEED)
    return s


# =============================================================================
section("1. ACTIVITY MODEL (Davies / Debye-Huckel)")
# =============================================================================
A25 = dh_A(25.0)
ck("Debye-Huckel A at 25 C", A25, 0.509, 0.004, "(mol/L)^-0.5")
ck("gamma(z=1) at I = 0 is exactly 1", davies_gamma(1, 0.0, A25), 1.0, 0.0)
ck("gamma(z=2) at I = 0 is exactly 1", davies_gamma(2, 0.0, A25), 1.0, 0.0)
ck("gamma(z=1) at I = 0.001", davies_gamma(1, 1e-3, A25), 0.9643, 0.002)
ck("gamma(z=2) at I = 0.001", davies_gamma(2, 1e-3, A25), 0.8683, 0.004)
ck_true("gamma(2+) < gamma(1+) at every I (charge dependence)",
        all(davies_gamma(2, I, A25) < davies_gamma(1, I, A25)
            for I in (1e-4, 1e-3, 1e-2, 5e-2)))
ck_true("gamma decreases monotonically with I",
        all(davies_gamma(2, a, A25) > davies_gamma(2, b, A25)
            for a, b in zip([1e-4, 1e-3, 5e-3], [1e-3, 5e-3, 2e-2])))
ck_true("A increases with temperature (dielectric falls)",
        dh_A(5) < dh_A(25) < dh_A(45), f"{dh_A(5):.4f} {dh_A(25):.4f} {dh_A(45):.4f}")

# =============================================================================
section("2. EQUILIBRIUM CONSTANTS")
# =============================================================================
K0 = conditional_constants(25.0, 0.0)
ck("pK1' at I=0 equals thermodynamic pK1", -math.log10(K0["K1"]), 6.35, 1e-9)
ck("pK2' at I=0 equals thermodynamic pK2", -math.log10(K0["K2"]), 10.33, 1e-9)
ck("pKw' at I=0 equals thermodynamic pKw", -math.log10(K0["Kw"]), 14.0, 1e-9)
ck("pKsp' at I=0 equals thermodynamic pKsp", -math.log10(K0["Ksp"]), 8.48, 1e-9)
K1 = conditional_constants(25.0, 0.01)
ck_true("conditional constants all shift DOWN as I rises",
        all(-math.log10(K1[k]) < -math.log10(K0[k]) for k in ("K1", "K2", "Kw", "Ksp")))
ck_true("K_H is not activity-corrected (CO2 is uncharged)",
        K1["KH"] == K0["KH"])
T = thermo_constants
ck_true("Van't Hoff: K1 rises with T (endothermic, dH>0)", T(40)["K1"] > T(10)["K1"])
ck_true("Van't Hoff: Ksp falls with T (exothermic, dH<0)", T(40)["Ksp"] < T(10)["Ksp"])
ck_true("Van't Hoff: K_H falls with T (gas less soluble when hot)",
        T(40)["KH"] < T(10)["KH"])
ck_true("Van't Hoff: Kw rises with T", T(40)["Kw"] > T(10)["Kw"])

# =============================================================================
section("3. pH SOLVER")
# =============================================================================
mono = True
for CT in (1e-6, 5e-4, 2e-3, 1e-2):
    prev = None
    for i in range(300):
        H = 10 ** (-(1.0 + i * 13.0 / 299))      # [H+] descending
        v = alk_from_H(H, CT, K0)
        if prev is not None and v <= prev:
            mono = False
        prev = v
ck_true("alkalinity is strictly monotonic in [H+] (root is unique)", mono)

worst = 0.0
for CT in (1e-6, 5e-4, 2e-3, 1e-2):
    for Alk in (-1e-4, 0.0, 1e-5, 1e-3, 5e-3):
        h = solve_H(Alk, CT, K0)
        if not math.isnan(h):
            worst = max(worst, abs(alk_from_H(h, CT, K0) - Alk))
ck("solver root satisfies the alkalinity balance", worst, 0.0, 1e-12, "eq/L")

# round-trip through the three specification modes
Ca, Mg, Cl, Tc = 1.5e-3, 4e-4, 1.2e-3, 22.0
Alk_t, CT_t = 1.4e-3, 1.6e-3
st_ = build_state("x", Ca, Mg, CT_t, Alk_t, Cl, Tc)
ck("CT_from_pH_alk recovers C_T",
   CT_from_pH_alk(st_["pH"], Alk_t, Ca, Mg, Cl, Tc) * 1e3, CT_t * 1e3, 1e-6, "mmol/L")
ck("alk_from_pH_CT recovers alkalinity",
   alk_from_pH_CT(st_["pH"], CT_t, Ca, Mg, Cl, Tc) * 1e3, Alk_t * 1e3, 1e-6, "meq/L")

# =============================================================================
section("4. REDUCTION TO THE ORIGINAL MODEL (I -> 0)")
# =============================================================================
try:
    O = _load_model("app_co2_units_fixed.py", "# SVG PROCESS UNIT ILLUSTRATIONS")
    worst_ph = worst_si = 0.0
    for Ca_mg, Mg_mg, alk_mg, CTv, Tv in [(30, 5, 60, 1.1e-3, 25.), (60, 10, 120, 2.0e-3, 25.),
                                          (100, 20, 200, 3.4e-3, 25.), (5, 1, 20, 5e-4, 10.),
                                          (80, 15, 150, 2.5e-3, 45.), (200, 40, 400, 6e-3, 30.)]:
        ca, mg, alk = Ca_mg / (MW["Ca"] * 1e3), Mg_mg / (MW["Mg"] * 1e3), alk_mg / 5e4
        old = O["state_basic"]("o", ca, mg, CTv, alk, Tv)
        Kc = conditional_constants(Tv, 0.0)
        H = solve_H(alk, CTv, Kc)
        _, _, _, _, _, co3 = speciate(H, CTv, Kc)
        worst_ph = max(worst_ph, abs(-math.log10(H) - old["pH"]))
        worst_si = max(worst_si, abs(math.log10(ca * co3 / Kc["Ksp_T"]) - old["SI"]))
    ck("pH matches the original exactly at I=0", worst_ph, 0.0, 1e-9)
    ck("SI matches the original exactly at I=0", worst_si, 0.0, 1e-9)
except FileNotFoundError:
    print("  SKIP  original app not present for comparison")

# =============================================================================
section("5. TEXTBOOK BENCHMARK: calcite in equilibrium with the atmosphere")
# =============================================================================
from scipy.optimize import brentq
pco2 = 10 ** -3.5


def _closure(pH, Tv=25.0):
    I = 0.0
    for _ in range(60):
        Kc = conditional_constants(Tv, I)
        H = 10 ** (-pH) / Kc["g1"]
        den = H * H + Kc["K1"] * H + Kc["K1"] * Kc["K2"]
        CTv = (Kc["KH"] * pco2) / (H * H / den)
        _, _, _, _, hco3, co3 = speciate(H, CTv, Kc)
        alk = hco3 + 2 * co3 + Kc["Kw"] / H - H
        ca = alk / 2
        I = 0.5 * (4 * ca + H + Kc["Kw"] / H + hco3 + 4 * co3)
    return math.log10((Kc["g2"] * ca) * (Kc["g2"] * co3) / Kc["Ksp_T"]), ca, alk, CTv, I


pH_eq = brentq(lambda p: _closure(p)[0], 6.0, 11.0, xtol=1e-10)
si_eq, ca_eq, alk_eq, ct_eq, I_eq = _closure(pH_eq)
ck("equilibrium pH (textbook 8.3)", pH_eq, 8.30, 0.15)
ck("equilibrium Ca (textbook ~20 mg/L)", ca_eq * MW["Ca"] * 1e3, 20.1, 2.5, "mg/L")
ck("equilibrium alkalinity (textbook ~1.0)", alk_eq * 1e3, 1.0, 0.15, "meq/L")
ck("residual SI at that point is zero", si_eq, 0.0, 1e-6)
s_rt = build_state("rt", ca_eq, 0.0, ct_eq, alk_eq, 0.0, 25.0)
ck("feeding it back through build_state gives SI=0", s_rt["SI"], 0.0, 1e-6)
ck("...and the same pH", s_rt["pH"], pH_eq, 1e-6)

# =============================================================================
section("6. INTERNAL CONSISTENCY")
# =============================================================================
zero = {"CaOH2": 0.0, "CO2": 0.0, "CaCO3": 0.0, "MgCl2": 0.0}
worst = 0.0
for pH in (6.0, 6.5, 7.0, 7.5, 8.0):
    kw = dict(FEED); kw["pH0"] = pH
    f, _, _ = make_initial(**kw)
    worst = max(worst, abs(remineralize(f, zero, False)["pH"] - pH))
ck("adding nothing does not change the water", worst, 0.0, 1e-9, "pH")

a, _, _ = make_initial("pH + alkalinity", 7.20, 45.0, 5e-4, 5., 1., 60., 25.)
b, _, _ = make_initial("pH + Cₜ", 7.20, 45.0, a["CT"], 5., 1., 60., 25.)
c, _, _ = make_initial("alkalinity + Cₜ", 7.20, a["Alk_mg"], a["CT"], 5., 1., 60., 25.)
ck("mode 'pH+Ct' agrees with 'pH+Alk'", b["pH"], a["pH"], 1e-7, "pH")
ck("mode 'Alk+Ct' agrees with 'pH+Alk'", c["pH"], a["pH"], 1e-7, "pH")
ck("...and on alkalinity", b["Alk_mg"], a["Alk_mg"], 1e-6, "mg/L")
ck("...and on C_T", c["CT"] * 1e3, a["CT"] * 1e3, 1e-9, "mmol/L")

# CCPP is defined as the amount that brings SI to zero - so verify it does
init = feed()
for label, dz in [("oversaturated", DOSES),
                  ("aggressive", {"CaOH2": 2., "CO2": 40., "CaCO3": 0., "MgCl2": 5.})]:
    r = remineralize(init, dz, True)
    y = r["CCPP"]
    after = build_state("after", r["Ca"] - y, r["Mg"], r["CT"] - y, r["Alk"] - 2 * y,
                        r["Cl"], r["T"])
    ck(f"removing CCPP drives SI to zero ({label})", after["SI"], 0.0, 1e-6)
    ck_true(f"CCPP sign matches SI sign ({label})",
            (y > 0) == (r["SI"] > 0), f"SI={r['SI']:+.3f} CCPP={y:+.3g}")

# =============================================================================
section("7. MASS BALANCES THROUGH THE PROCESS")
# =============================================================================
init = feed()
rem = remineralize(init, DOSES, False)
n_lime = DOSES["CaOH2"] / (MW["CaOH2"] * 1e3)
n_cal = DOSES["CaCO3"] / (MW["CaCO3"] * 1e3)
n_co2 = DOSES["CO2"] / (MW["CO2"] * 1e3)
n_mg = DOSES["MgCl2"] / (MW["MgCl2"] * 1e3)
ck("dosing: Ca in = Ca(OH)2 + CaCO3", rem["Ca"] - init["Ca"], n_lime + n_cal, 1e-15, "mol/L")
ck("dosing: C_T in = CO2 + CaCO3", rem["CT"] - init["CT"], n_co2 + n_cal, 1e-15, "mol/L")
ck("dosing: alkalinity in = 2(lime + calcite)",
   rem["Alk"] - init["Alk"], 2 * (n_lime + n_cal), 1e-15, "eq/L")
ck("dosing: MgCl2 contributes 2 Cl per Mg", rem["Cl"] - init["Cl"], 2 * n_mg, 1e-15, "mol/L")

res, _ = reservoir(rem, 2.0, 1000., 100., 25., False, 60)
ck("reservoir conserves calcium", res["Ca"], rem["Ca"], 0.0, "mol/L")
ck("reservoir conserves alkalinity", res["Alk"], rem["Alk"], 0.0, "eq/L")
out, _, px = pipe(res, 5000., 0.5, 500., 30., False, 80, False)
ck("pipe: Ca removed equals CaCO3 deposited", res["Ca"] - out["Ca"], px["precip"], 1e-15, "mol/L")
ck("pipe: C_T removed equals CaCO3 deposited", res["CT"] - out["CT"], px["precip"], 1e-15, "mol/L")
ck("pipe: alkalinity removed equals 2x deposited",
   res["Alk"] - out["Alk"], 2 * px["precip"], 1e-15, "eq/L")
ck("pipe conserves magnesium", out["Mg"], res["Mg"], 0.0, "mol/L")

# =============================================================================
section("8. PROCESS UNITS BEHAVE PHYSICALLY")
# =============================================================================
c2, _ = reservoir(rem, 2.0, 1000., 100., 25., False, 60, False)
c72, _ = reservoir(rem, 72.0, 1000., 100., 25., False, 60, False)
ck_true("closed reservoir equilibrates and then stops changing",
        abs(c72["CT"] - c2["CT"]) / c2["CT"] < 1e-4,
        f"{c2['CT']:.6g} -> {c72['CT']:.6g}")
ck_true("closed-mode result is step-size independent",
        abs(reservoir(rem, 72., 1000., 100., 25., False, 1500, False)[0]["CT"]
            - c72["CT"]) / c72["CT"] < 1e-5,
        "coarse integration must not shift the answer")

v24, _ = reservoir(rem, 24.0, 1000., 100., 25., False, 60, True)
v72, _ = reservoir(rem, 72.0, 1000., 100., 25., False, 60, True)
ck_true("vented reservoir keeps exchanging with time", v72["CT"] > v24["CT"] > rem["CT"])
KH25 = thermo_constants(25.0)["KH"]
vlong, _ = reservoir(rem, 2000.0, 1000., 100., 25., False, 600, True)
slong = build_state("v", vlong["Ca"], vlong["Mg"], vlong["CT"], vlong["Alk"], vlong["Cl"], 25.)
ck("vented reservoir settles at Henry equilibrium",
   slong["CO2"] / (KH25 * M["PCO2_GAS_0"]), 1.0, 1e-3, "ratio")
a_, b_ = reservoir(rem, 72., 1000., 100., 25., False, 60, True)[0], \
         reservoir(rem, 72., 1000., 100., 25., False, 600, True)[0]
ck("vented result is step-size converged", a_["SI"], b_["SI"], 5e-3, "SI")

hot, _, _ = pipe(res, 5000., 0.5, 500., 5.0, False, 80, False)
ins, _, _ = pipe(res, 5000., 0.5, 500., 5.0, True, 80, False)
ck_true("insulated pipe loses less heat", ins["T"] > hot["T"],
        f"exposed {hot['T']:.2f} C vs insulated {ins['T']:.2f} C")
lng, _, plx = pipe(res, 200000., 0.5, 500., 30., False, 200, False)
ck_true("a very long pipe drives the water toward equilibrium",
        abs(lng["SI"]) < abs(res["SI"]), f"{res['SI']:+.3f} -> {lng['SI']:+.3f}")
ck_true("deposition only occurs when oversaturated", plx["precip"] >= 0.0)
under, _, upx = pipe(build_state("u", 1e-5, 1e-5, 5e-4, 1e-4, 1e-3, 25.),
                     5000., 0.5, 500., 30., False, 80, False)
ck("undersaturated water deposits nothing", upx["precip"], 0.0, 0.0, "mol/L")
fast, _, fx = pipe(res, 5000., 0.5, 1000., 30., False, 80, False)
slow, _, sx = pipe(res, 5000., 0.5, 250., 30., False, 80, False)
ck_true("slower flow -> longer contact -> more deposition", sx["precip"] > fx["precip"],
        f"{sx['precip']:.3g} vs {fx['precip']:.3g}")

# =============================================================================
section("9. CROSS-CHECK vs PHREEQC (independent reference engine)")
# =============================================================================
try:
    import phreeqpython
    pp = phreeqpython.PhreeqPython()
    waters = [("soft", 8.00, 50., 25., 5., 60.), ("typical", 8.20, 80., 40., 8., 60.),
              ("hard", 7.80, 120., 60., 12., 100.), ("aggressive", 6.80, 15., 5., 1., 40.)]
    errs = []
    for nm, pH, alk, cam, mgm, nacl in waters:
        sol = pp.add_solution_raw({
            "temp": "25", "pH": str(pH), "units": "mg/l", "Ca": str(cam), "Mg": str(mgm),
            "Alkalinity": f"{alk} as CaCO3",
            "Na": str(nacl * 22.98977 / MW["NaCl"]), "Cl": "1 charge"})
        ours, _, _ = make_initial("pH + alkalinity", pH, alk, 5e-4, cam, mgm, nacl, 25.)
        errs.append(abs(ours["SI"] - sol.si("Calcite")))
    ck("mean |SI error| vs PHREEQC", sum(errs) / len(errs), 0.0, 0.08, "log units")
    ck("worst |SI error| vs PHREEQC", max(errs), 0.0, 0.10, "log units")
except ImportError:
    print("  SKIP  phreeqpython not installed (pip install phreeqpython)")

# =============================================================================
print("\n" + "=" * 72)
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 72)
sys.exit(FAIL)
