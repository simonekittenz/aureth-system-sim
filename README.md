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

## Two pages

- **Bayurinda moons** (`moons.py`): Liwaya and Nineya, their phases, tides, calendar, eclipses and stability.
- **Aureth system** (`system.py`): the star with Bayurinda, Ashara and Tarnic: orbits, cycles and alignments, Bayurinda's
  sky, Sternreyker flight times between planets (constant acceleration, 0.6 to 1.4 g), a ladder of long cycles, and long-run stability. The other planets join once they have a distance.

`app.py` is only the router between them.

## Layout

| File | Purpose |
|---|---|
| `bajurinda/constants.py` | Physical constants |
| `bajurinda/model.py` | Planet and moon definitions, derived properties, Hill/Roche limits, tide scaling, warnings |
| `bajurinda/presets.py` | Candidate moon configurations (hypothetical, not canon) |
| `bajurinda/dynamics.py` | REBOUND N-body integration, stability runs, random-phase ensembles |
| `bajurinda/sky.py` | Phases, altitude, equilibrium tides, alignment events |
| `bajurinda/plots.py` | Plotly figures for the moons page |
| `bajurinda/aureth.py` | The star and planets: loading `data/aureth.yaml`, orbits, N-body, sky from Bayurinda, alignments |
| `bajurinda/flights.py` | Sternreyker flights: constant-acceleration flip-and-burn, Sun keep-out detours, best/worst timing |
| `bajurinda/landmarks.py` | Horizon alignments: solstice sunrises, moon standstills, return cycles, star drift |
| `bajurinda/cycles.py` | Long cycles: eclipse cycles, node turns, plane cycle, planet repeats, the Great Cycle, axis precession |
| `bajurinda/transfers.py` | Coasting comparison (no wings): Lambert solver, Hohmann baseline, window finder |
| `bajurinda/system_plots.py` | Plotly figures for the system page |
| `app.py`, `moons.py`, `system.py` | The UI: router and the two pages |

## Calendar tab

Lists every month (Liwaya's phase cycle) and week (Nineya's phase cycle) across the repeating cycle, with
runs of double-full and double-new moons marked. Month and week names and structure are placeholders.

## Eras tab

A long-run timeline (70-300 years) of how often the two moons overlap, the angle between their orbital planes,
and where the double-full and double-new months fall. Gold bands mark the stretches when the planes are aligned.

## Night sky tab

The sky from a chosen latitude, longitude and local time: Aureth, both moons with their true phases, Ashara and Tarnic, the
horizon with compass points, a set of named bright stars and a Milky Way. The stars and the galaxy's shape come from
the `sky` section of `data/aureth.yaml` and are all placeholders: rename and move them freely. A panorama or an all-sky dome, with a
Play button that runs a time-lapse. The picture sits at the top of the tab, its height is a setting (up to 1500 px), and the
full-screen button at its top right goes larger still. You choose the start, the time between frames (5 minutes to 10 days), how long it plays
(an hour to four years) and the speed. The founding moment is the default view.

### Reading coordinates off the sky

Tick **Pointer coordinates** in the Night sky tab. The sky becomes a still picture of the start moment, and hovering shows the ecliptic
longitude and latitude under the pointer (the same numbers the `sky` section uses), plus the height, direction and declination, and the
nearest named star. A **Preview a star** box lets you type coordinates and see where a star would sit, and gives the line to paste into the
data file. Narrow the panorama width for finer pointing.

### Editing the sky

In `data/aureth.yaml`, under `sky`:

- `stars`: each has a `name`, `lon` and `lat` (ecliptic coordinates; longitude 180 is where Aureth sits at the founding moment), a `mag`
  (smaller is brighter), a `color` and whether to show its name.
- `faint_stars`: the `seed` and counts for the generated dim stars. The same seed always gives the same sky. The sidebar has a
  seed box, a button for a new random seed, one to go back to the saved seed, and one that writes the current seed into this file.
- `milky_way`: the `pole` of the galaxy's plane, `centre_lon` (where the bright centre lies), `width_deg`, `bulge_width_deg` and
  `bulge_length_deg` (how much it swells toward the centre), a dark `rift` and any number of `patches` (positive glows, negative dark clouds).

## Landmarks tab

Where the Sun and each moon rise on the horizon at a chosen latitude, the moons' standstills (extreme rising points), cycles after
which the same phase, hour, height and date return, and how long an alignment to a fixed star stays true.

## Eclipses tab

Solar and lunar eclipses for both moons, counted for the whole planet and for a single place, plus a table of what
different star sizes would do to totality.

## Not yet implemented

Tidal evolution (REBOUNDx), tidal locking timescales, precession.

## Design decisions

- The 19 h day is a **solar day** (noon to noon). Liwaya is the **larger** moon, Nineya the **smaller** (per the world files).
- **Aureth** is a G-type star, "the golden Aureth": 1.0021 solar masses (pinned by the original canon), 0.985 solar radii,
  0.87 solar luminosities, about 5,620 K. This matches a mature (about 5 Gyr), mildly metal-rich ([Fe/H] about +0.14)
  star in the MIST v1.2 tracks, calibrated to the real Sun. It is a proposal; radius and luminosity are free to change.
- **Bayurinda** sits at about 0.97 AU with a year of exactly 440 solar days (the original canon was 0.87 AU and 374 days),
  so the Sun looks 0.541 degrees wide and Liwaya (0.545 degrees) can fully cover it. Sunlight is 0.93x Earth's.
  Tarnic is unchanged (the star's mass is unchanged).
- Moon configuration **B** is the selected design (hypothetical, not yet canon):
  - Liwaya: 3,450 km across, about 362,900 km from Bayurinda on average (28.0 days sidereal, **38.5 Bajurindan days** per phase cycle).
  - Nineya: 1,200 km across, about 150,250 km on average (7.5 days sidereal, **9.70 Bajurindan days** per phase cycle).
  - 32 Liwaya cycles = 127 Nineya cycles = 1,232 days (2.80 years). The calendar is deliberately not tied to the solar year.
  - Both orbits are tilted 1.5 degrees and start at a double full moon with the two moons overlapping.
  - Stable over 3,000 simulated years and across random starting phases.
  - The orbital planes drift apart and realign every 52.7 Bajurindan years; overlaps are frequent for a few years around each
    realignment (about 5 a year on average).
  - Distances in the sidebar are starting (osculating) values; the Overview shows the simulated averages.

## Calendar tab

Lists every month (Liwaya's phase cycle) and week (Nineya's phase cycle) across the repeating cycle, with
runs of double-full and double-new moons marked. Month and week names and structure are placeholders.

## Eras tab

A long-run timeline (70-300 years) of how often the two moons overlap, the angle between their orbital planes,
and where the double-full and double-new months fall. Gold bands mark the stretches when the planes are aligned.

## Night sky tab

The sky from a chosen latitude, longitude and local time: Aureth, both moons with their true phases, Ashara and Tarnic, the
horizon with compass points, a set of named bright stars and a Milky Way. The stars and the galaxy's shape come from
the `sky` section of `data/aureth.yaml` and are all placeholders: rename and move them freely. A panorama or an all-sky dome, with a
Play button that runs a time-lapse. You choose the start, the time between frames (5 minutes to 10 days), how long it plays
(an hour to four years) and the speed. The founding moment is the default view.

## Landmarks tab

Where the Sun and each moon rise on the horizon at a chosen latitude, the moons' standstills (extreme rising points), cycles after
which the same phase, hour, height and date return, and how long an alignment to a fixed star stays true.

## Eclipses tab

Solar and lunar eclipses for both moons, counted for the whole planet and for a single place, plus a table of what
different star sizes would do to totality.

## Not yet implemented

Tidal evolution (REBOUNDx), tidal locking timescales, precession.

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
- Aureth is a **G-type** star: 1.0021 solar masses (derived from the 374-day year at 0.87 AU). Its radius (0.88 solar) and
  luminosity (0.72 solar) are a proposal for a young star: sunlight at Bayurinda is 0.95x Earth, and the Sun looks small
  enough (0.539 degrees) that Liwaya (0.55 degrees) can fully cover it for total solar eclipses.
