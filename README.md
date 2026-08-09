# Remineralization & Water Transport Simulator

Interactive model of the post-treatment train that turns desalinated permeate into
distributed drinking water, and of the calcium-carbonate stability of that water
from the plant to the tap.

```
Desalinated water → Remineralization → Closed reservoir → Supply pipe → Consumer
```

## What the model contains

| Element | Implementation |
|---|---|
| Carbonate equilibrium | H₂CO₃* / HCO₃⁻ / CO₃²⁻, alkalinity-based pH solution |
| Temperature | Van't Hoff correction of K₁, K₂, K_w, K_sp, K_H |
| **Ionic strength** | Davies activity coefficients → conditional equilibrium constants, iterated to self-consistency with the speciation |
| **Mass transfer** | CO₂ exchange with a finite closed headspace, kLa driving force toward Henry's-law equilibrium |
| **Kinetics** | SI-driven CaCO₃ deposition along the pipe, with simultaneous heat loss |
| Stability indices | Saturation Index (from ion **activities**) and CCPP |

pH is reported on the activity scale, `pH = −log₁₀(γ₁[H⁺])`. At `I = 0` the model
reduces exactly to the infinite-dilution case.

## Two versions, switchable

The original project is kept in the repository, unmodified, alongside the
enhanced build. There are three possible entry points:

| Entry point | Serves |
|---|---|
| `streamlit_app.py` | the enhanced version only |
| `app_co2_units_fixed.py` | the original version only |
| `launcher.py` | **both**, switchable at runtime from the sidebar |

`launcher.py` runs whichever version you pick without altering either file, and
writes the choice to the URL (`?app=original`) so a specific version can be
linked to. Use it if you want one deployment that can be flipped back to the
original at any time.

## Running it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m streamlit run launcher.py          # both, switchable
.venv/bin/python -m streamlit run streamlit_app.py     # enhanced only
.venv/bin/python -m streamlit run app_co2_units_fixed.py   # original only
```

## Deploying to Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick this repository.
3. Set **Main file path** to `launcher.py` (or either single-version file).
4. **Deploy.**

`requirements.txt` and `.streamlit/config.toml` are already set up; no secrets
are required. The first build takes 2–3 minutes.

**Switching which version is live** — either flip the sidebar control if you
deployed `launcher.py`, or change *Main file path* in the app's settings on
Streamlit Cloud and let it redeploy. No code changes either way.

## Verifying the model

```bash
.venv/bin/python validation.py        # 61 checks; exit code = failures
.venv/bin/python fingerprint.py --check   # confirms no computed value drifted
```

`validation.py` covers the activity model, the equilibrium constants, the pH
solver, exact reduction to the original model at zero ionic strength, the
textbook calcite/atmosphere benchmark, internal consistency, mass balances,
process-unit behaviour, and a cross-check against PHREEQC where
`phreeqpython` is installed.

## A note on the feed water

pH, alkalinity and total inorganic carbon (C⊤) are linked by the carbonate
equilibrium — only **two** are independent. The sidebar therefore asks for two and
calculates the third, and reports which one it derived. Specifying all three
independently is what allows a model to display one water while simulating a
different one.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | The application (single file — the assignment requires the code as one submittable listing) |
| `simulator_code.txt` | Identical to the above, renamed for submission |
| `requirements.txt` | Pinned minimum dependency versions |
| `.streamlit/config.toml` | Theme and server configuration |
| `app_co2_units_fixed.py` | The original version, kept for comparison |
