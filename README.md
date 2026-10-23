# Bajurinda two-moon simulator

Interactive Streamlit app for designing Liwaya and Nineya, the moons of Bajurinda (*Alteria*).

## Run

```bash
cd ~/Projects/Creative/planet-sims
source .venv/bin/activate
streamlit run app.py
```

A browser tab opens at http://localhost:8501.

## Layout

| File | Purpose |
|---|---|
| `bajurinda/constants.py` | Physical constants |
| `bajurinda/model.py` | Planet and moon definitions, derived properties, Hill/Roche limits, tide scaling, warnings |
| `bajurinda/presets.py` | Candidate moon configurations (hypothetical, not canon) |
| `bajurinda/dynamics.py` | REBOUND N-body integration, stability runs, random-phase ensembles |
| `bajurinda/sky.py` | Phases, altitude, equilibrium tides, alignment events |
| `bajurinda/plots.py` | Plotly figures |
| `app.py` | The UI |

## Not yet implemented

Tidal evolution (REBOUNDx), tidal locking timescales, eclipses, precession.

## Design decisions

- The 19 h day is a **solar day** (noon to noon). Sidereal rotation is therefore 18.949 h, and a year is 374 solar days or 375 rotations.
- Liwaya is the **larger** moon and Nineya the **smaller** (per the world files).
- Moon configuration **B** is the selected design (hypothetical, not yet canon):
  - Liwaya: 3,450 km across, about 359,000 km from Bayurinda on average (28 days sidereal, 38.36 Bajurindan days per phase cycle).
  - Nineya: 1,200 km across, about 149,400 km on average (7.4 days sidereal, 9.65 Bajurindan days per phase cycle).
  - Orbits are tuned so that **39 Liwaya cycles = 155 Nineya cycles = 4 Bajurindan years**.
  - Both orbits are tilted 1.5 degrees and start at a double full moon with the two moons overlapping.
  - Stable over 3,000 simulated years and across random starting phases.
  - The orbital planes drift apart and realign about every 66.5 Bajurindan years, so overlaps are frequent for a few
    years around each realignment and rarer in between.
  - Distances in the sidebar are starting (osculating) values; the Overview shows the simulated averages.
- Aureth is a **G-type** star: about 1.00 solar masses (derived from the 374-day year at 0.87 AU) and solar luminosity.
