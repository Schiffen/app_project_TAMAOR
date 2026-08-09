"""
Numerical fingerprint of the simulator.
=======================================

Sweeps the model over a wide grid of inputs and hashes every computed value.
Design work must never change this hash.

    python fingerprint.py            # print the fingerprint
    python fingerprint.py --save     # write fingerprint.json
    python fingerprint.py --check    # compare against fingerprint.json, exit 1 on drift
"""
import hashlib
import json
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Stub:
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return self
    def __contains__(self, x): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def setdefault(self, *a): return None


def _load():
    m = types.ModuleType("streamlit")
    m.__getattr__ = lambda n: _Stub()
    sys.modules["streamlit"] = m
    path = os.path.join(_HERE, "streamlit_app.py")
    src = open(path).read().split("# 8. PROCESS-UNIT ILLUSTRATIONS")[0]
    src = (src.replace("st.set_page_config(", "_stub(")
              .replace("st.markdown(", "_stub(")
              .replace("@st.cache_data(show_spinner=False, max_entries=512)", ""))
    ns = {"_stub": lambda *a, **k: None}
    exec(compile(src, path, "exec"), ns)
    return ns


M = _load()

FIELDS = ("pH", "SI", "I", "g1", "g2", "Alk_mg", "Ca_mg", "Mg_mg", "TH",
          "CT", "Cl", "CO2", "HCO3", "CO3", "a0", "a1", "a2", "T")


def _row(s):
    return [round(float(s[k]), 10) if s[k] == s[k] else "nan" for k in FIELDS]


def sweep():
    """Every value the model produces across a broad input grid."""
    out = []
    modes = ["pH + alkalinity", "pH + Cₜ", "alkalinity + Cₜ"]
    for mode in modes:
        for pH0 in (6.0, 6.5, 7.4, 8.1):
            for alk0 in (10.0, 20.0, 90.0):
                for T0 in (8.0, 25.0, 42.0):
                    for nacl in (0.0, 60.0, 400.0):
                        init, derived, problem = M["make_initial"](
                            mode, pH0, alk0, 5e-4, 5.0, 1.0, nacl, T0)
                        if init is None:
                            out.append(["feed-rejected", mode, pH0, alk0, T0, nacl])
                            continue
                        out.append(_row(init))
                        out.append([round(float(derived[2]), 10)
                                    if derived[2] == derived[2] else "nan"])

    base, _, _ = M["make_initial"]("pH + alkalinity", 6.5, 20.0, 5e-4, 5.0, 1.0, 60.0, 25.0)
    for lime in (0.0, 12.0, 55.0, 140.0):
        for co2 in (0.0, 15.0, 70.0):
            for cal in (0.0, 30.0, 180.0):
                dz = {"CaOH2": lime, "CO2": co2, "CaCO3": cal, "MgCl2": 5.0}
                rem = M["remineralize"](base, dz, True)
                out.append(_row(rem) + [round(float(rem["CCPP"]), 12)
                                        if rem["CCPP"] == rem["CCPP"] else "nan"])

                for vented in (False, True):
                    for t in (0.0, 2.0, 30.0):
                        res, rx = M["reservoir"](rem, t, 1000.0, 100.0, 25.0,
                                                 True, 60, vented)
                        out.append(_row(res) + [round(float(rx["pCO2"]), 14)])

                        po, prof, px = M["pipe"](res, 5000.0, 0.5, 500.0, 30.0,
                                                 False, 80, True)
                        out.append(_row(po) + [round(float(px["precip"]), 14),
                                               round(float(px["v"]), 12),
                                               round(float(px["t_s"]), 8)])
                        out.append([round(float(prof["SI"].iloc[i]), 10)
                                    for i in (0, 20, 40, 60, 80)])
    return out


def fingerprint():
    rows = sweep()
    blob = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest(), len(rows)


if __name__ == "__main__":
    digest, n = fingerprint()
    store = os.path.join(_HERE, "fingerprint.json")

    if "--save" in sys.argv:
        json.dump({"sha256": digest, "rows": n}, open(store, "w"), indent=2)
        print(f"saved {n} rows -> {digest}")
    elif "--check" in sys.argv:
        if not os.path.exists(store):
            print("no fingerprint.json to compare against; run --save first")
            sys.exit(2)
        ref = json.load(open(store))
        same = ref["sha256"] == digest and ref["rows"] == n
        print(f"  reference : {ref['sha256']}  ({ref['rows']} rows)")
        print(f"  current   : {digest}  ({n} rows)")
        print("  MATCH — no computed value changed." if same
              else "  DRIFT — the model output changed!")
        sys.exit(0 if same else 1)
    else:
        print(f"{n} rows -> {digest}")
