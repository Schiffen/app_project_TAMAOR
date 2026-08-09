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

## Entry points

The deployed app serves the enhanced version and only that — there is no
control for switching versions. The original project is still kept in the
repository, unmodified, and still runs on its own if you want to compare them
locally.

| Entry point | Serves |
|---|---|
| `launcher.py` | the enhanced version (what the deployment runs) |
| `streamlit_app.py` | the enhanced version, run directly |
| `app_co2_units_fixed.py` | the original version, for comparison |

## Running it on your own computer

Needs Python 3.9 or newer. Nothing else — no database, no API keys, no account.

**macOS / Linux**

```bash
git clone https://github.com/Schiffen/app_project_TAMAOR.git
cd app_project_TAMAOR
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run launcher.py
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Schiffen/app_project_TAMAOR.git
cd app_project_TAMAOR
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamlit run launcher.py
```

A browser tab opens at `http://localhost:8501` automatically. If it doesn't,
open that address yourself. Press `Ctrl+C` in the terminal to stop it.

The first install takes a couple of minutes (it downloads numpy, scipy, pandas,
plotly and streamlit). After that, startup is a few seconds.

To run the original build for comparison, replace `launcher.py` with
`app_co2_units_fixed.py`.

### No Python? 

Download the repository as a ZIP from the green **Code** button on GitHub, or
just open the deployed link if one has been published — that needs nothing
installed at all.

## Deploying to Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick this repository.
3. Set **Main file path** to `launcher.py` (or `streamlit_app.py` — both serve
   the enhanced version).
4. **Deploy.**

`requirements.txt` and `.streamlit/config.toml` are already set up; no secrets
are required. The first build takes 2–3 minutes.

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

## A note on the desalinated water

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
