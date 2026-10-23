# Bajurinda two-moon simulator

Interactive Streamlit app for designing Liwaya and Nineya, the moons of Bajurinda (*Alteria*).

## Run

Shortcut (set up in `~/.zshrc`; open a new Terminal window first):

```bash
bajurinda            # start, or restart if already running
bajurinda stop
bajurinda status
```

Without the shortcut:

```bash
cd ~/Projects/Creative/planet-sims
source .venv/bin/activate
streamlit run app.py
```

A browser tab opens at http://localhost:8501. Keep the Terminal window open while you use the app.

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

## Calendar tab

Lists every month (Liwaya's phase cycle) and week (Nineya's phase cycle) across the repeating cycle, with
runs of double-full and double-new moons marked. Month and week names and structure are placeholders.

## Eras tab

A long-run timeline (70-300 years) of how often the two moons overlap, the angle between their orbital planes,
and where the double-full and double-new months fall. Gold bands mark the stretches when the planes are aligned.

## Not yet implemented

Tidal evolution (REBOUNDx), tidal locking timescales, eclipse tracking, precession.

## Design decisions

- The 19 h day is a **solar day** (noon to noon). Sidereal rotation is therefore 18.949 h, and a year is 374 solar days or 375 rotations.
- Liwaya is the **larger** moon and Nineya the **smaller** (per the world files).
- Moon configuration **B** is the selected design (hypothetical, not yet canon):
  - Liwaya: 3,450 km across, about 359,500 km from Bayurinda on average (27.6 days sidereal, **38.5 Bajurindan days** per phase cycle).
  - Nineya: 1,200 km across, about 149,900 km on average (7.5 days sidereal, **9.70 Bajurindan days** per phase cycle).
  - 32 Liwaya cycles = 127 Nineya cycles = 1,232 days (3.29 years). The calendar is deliberately not tied to the solar year.
  - Both orbits are tilted 1.5 degrees and start at a double full moon with the two moons overlapping.
  - Stable over 3,000 simulated years and across random starting phases.
  - The orbital planes drift apart and realign every 66.4 Bajurindan years, so overlaps are frequent for a few years around
    each realignment and rarer in between.
  - Distances in the sidebar are starting (osculating) values; the Overview shows the simulated averages.
- Aureth is a **G-type** star: about 1.00 solar masses (derived from the 374-day year at 0.87 AU) and solar luminosity.
