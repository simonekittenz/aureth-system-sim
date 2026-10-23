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
- Moon configuration **B** is the selected design (hypothetical, not yet canon):
  - Liwaya: 1,200 km, 150,000 km out, 7.5-day sidereal orbit.
  - Nineya: 3,450 km, 360,000 km out, 27.7-day sidereal orbit.
  - Stable over 3,000 simulated years and across random starting phases.
