# Aureth system simulator

Interactive Streamlit app for the Aureth system in *Alteria*: the star, its planets, and Bayurinda's moons Liwaya and Nineya.

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

- **Planet view** (`planet_view.py`): pick a **point of view** (Bayurinda, Ashara or Tarnic) in the sidebar. You get that planet's
  properties, seasons (Landmarks) and Night sky, with the other planets in its sky. Bayurinda also has its moons: phases, tides,
  alignments, calendar, eras, eclipses and moon stability. Ashara and Tarnic have no moons in the world files yet, so those tabs are hidden.
- **Aureth system** (`system.py`): the star with Bayurinda, Ashara and Tarnic: orbits, cycles and alignments, Bayurinda's
  sky, Sternreyker flight times between planets (constant acceleration, 0.6 to 1.4 g), a ladder of long cycles, and long-run stability.

`app.py` is only the router between them, plus the **Reload data files** button.

### One model

Both pages use the same world (`bajurinda/world.py`): one star, three planets, and Bayurinda's two moons. The sidebar settings are shared,
so a change on one page shows on the other. Where positions are drawn they come from one REBOUND simulation: the Planet view runs the star,
all the planets and the chosen planet's moons together, and the Aureth system page uses the same planets (the moons count as part of
Bayurinda, since their tiny orbits would make thousand-year runs take hours). The planet and its moons are started so that their shared
centre of mass follows the planet's orbit; without that the moons' pull stretched Bayurinda's year by 0.16%, and the planets in the
night sky drifted from the Sun by half a degree a year. Quick formulas (resonances, the 75-year repeat, the coasting-flight comparison)
still use idealised two-body orbits and are labelled as such. Ashara's axial tilt and day length are **placeholders**.

## Layout

| File | Purpose |
|---|---|
| `bajurinda/constants.py` | Physical constants |
| `bajurinda/model.py` | Planet and moon definitions, derived properties, Hill/Roche limits, tide scaling, warnings |
| `bajurinda/world.py` | The whole system as one object (star, planets, moons); reads `data/aureth.yaml` |
| `bajurinda/worldstate.py` | The sidebar settings shared by both pages |
| `bajurinda/presets.py` | Candidate moon configurations (hypothetical, not canon) |
| `bajurinda/dynamics.py` | The REBOUND simulation (star, planets, moons), stability runs, random-phase ensembles |
| `bajurinda/sky.py` | Phases, altitude, equilibrium tides, alignment events |
| `bajurinda/plots.py` | Plotly figures for the moons page |
| `bajurinda/aureth.py` | The star and planets: orbits, the planets-only N-body and its ephemeris, sky from Bayurinda, alignments |
| `bajurinda/flights.py` | Sternreyker flights: constant-acceleration flip-and-burn, Sun keep-out detours, best/worst timing |
| `bajurinda/landmarks.py` | Horizon alignments: solstice sunrises, moon standstills, return cycles, star drift |
| `bajurinda/cycles.py` | Long cycles: eclipse cycles, node turns, plane cycle, planet repeats, the Great Cycle, axis precession |
| `bajurinda/transfers.py` | Coasting comparison (no wings): Lambert solver, Hohmann baseline, window finder |
| `bajurinda/system_plots.py` | Plotly figures for the system page |
| `app.py`, `planet_view.py`, `system.py` | The UI: router and the two pages |

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

### Constellation map

The Aureth system page has a **Constellation map** tab: the whole sky as two round charts (northern and southern), with your constellation sets drawn as
dashed lines, the bright stars, the faint stars and a soft Milky Way. It does not depend on which planet you stand on. The chart uses the sky's own
coordinates (ecliptic longitude around the edge, latitude as the rings). A search box finds a constellation or a bright star: the match is boxed and drawn
thick, the rest fade, and the chart switches to the side it is on. A table lists the matches with where they sit and their brightest star.

### Moon designs

The sidebar's **Moon design** picks the moons' settings. *Design B* is the settled design (nearly round orbits). *Design C* is an experiment for a
captured Liwaya: its orbit is eccentric (0.08) and tilted 2 degrees, so it swells and shrinks by about 6% either way each month; Liwaya is a little
denser and smaller (3.6 g/cm3) and Nineya a little lighter and bigger (3.0 g/cm3), each with the same mass as before. The starting distances were
retuned so the phase cycles stay at 38.5 and 9.70 days. The "How big each looks" chart in Phases & sky shows the swelling.
In the tests Design C stayed stable for 10,000 years; eccentricity 0.10 starts to fail some random starts and 0.15 breaks Nineya's orbit.
Liwaya's eccentricity also pumps Nineya's to about the same value, so Nineya swells and shrinks too.

### Local time or the same instant

**Time is** switches the clock. *Local time at the observer* (the default) follows you, as time zones do: the Sun is in the same place at the
same reading wherever you stand, so the stars barely change with longitude. *Same instant everywhere* uses the world clock at longitude 0,
so every longitude sees the same moment and moving east or west turns the planet under the sky. The labels then show the world clock and, in
brackets, the local time where you stand.

### Moving around and star-chart mode

Above the Night sky picture, the **West / North / South / East** buttons move your position on Bayurinda by the chosen step (1° to 45°),
changing the same latitude and longitude as the sidebar sliders. The readout shows where you stand and how far a degree is in km.
**Stars in daytime too** switches to a star-chart mode: stars, Milky Way and constellation lines stay visible whatever the Sun is doing,
and the sky background goes dark so they can be seen. Switch it off for the realistic sky.

### Reading coordinates off the sky

Tick **Pointer coordinates** in the Night sky tab. The sky becomes a still picture of the start moment, and hovering shows the ecliptic
longitude and latitude under the pointer (the same numbers the `sky` section uses), plus the height, direction and declination, and the
nearest named star. A **Preview a star** box lets you type coordinates and see where a star would sit, and gives the line to paste into the
data file. Narrow the panorama width for finer pointing.

### Reloading after you edit the data files

Streamlit re-runs when the Python files change, but it does not re-read the data folder, and the app also remembers its calculations.
After editing `data/aureth.yaml` or `data/constellations.yaml` by hand, press **Reload data files** at the top of the sidebar. It re-reads both,
clears the saved calculations, and leaves your sidebar settings alone. Unsaved constellation edits are kept (use *Undo unsaved changes* to
throw them away and take the file's version). If a file has a typo, the error is shown so you can fix it and press Reload again.

### Constellation sets

The Night sky tab has a **Constellation sets** box for joining the named stars into pictures. A *set* is one way of joining the stars
(one culture's pictures, say), a set holds *figures*, and a figure is a few lines, each joining two stars by name. Any number of sets can
sit on the same stars, and you can show several at once in different colours. You can create, duplicate (to rework the same figures into
another reading), rename and delete sets and figures, join stars with a chain, and remove lines. Edits show in the picture at once;
**Save** writes `data/constellations.yaml` (the file can also be edited by hand). Sets refer to stars by name, so if you rename a star in
`data/aureth.yaml`, rename it in the sets too (the app warns about lines whose stars no longer exist). The prefilled stars and
figures are placeholders.

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
