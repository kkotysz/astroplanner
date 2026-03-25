# Astronomical Observation Planner

A desktop GUI application for planning astronomical observations. It allows you to:

- Load celestial targets from a JSON or CSV/TSV file.
- Select an observation site and date.
- Calculate visibility curves for each target.
- Plot the results using Matplotlib.
- Display real-time altitudes, azimuths, sidereal time, and moon separation.

## Features

- Real-time updates of local, UTC, and sidereal time.
- Configurable observation site (latitude, longitude, elevation).
- Automatic sunrise/sunset, twilight, moonrise/moonset, and moon phase calculations.
- Ranking for each target (`Score`, hours above altitude limit).
- Observation filters (minimum moon separation, minimum score, hide observed targets).
- Extended target metadata (`type`, `magnitude`, `size`, `priority`, `observed`, `notes`).
- Best-effort SIMBAD lookup for missing `magnitude`.
- Export bundle: `plan_targets.json`, `plan_plot.png`, `plan_summary.csv`, `plan_schedule.ics`.
- Dark mode toggle.

## Installation

Clone the repository and change into its directory:

```bash
git clone <repo-url> astroplanner
cd astroplanner
```

### Conda

Create and activate the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate astroplanner
```

### Python venv

Create, activate, and install dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Run directly:

```bash
python astro_planner.py
```

Optionally load a plan on startup:

```bash
python astro_planner.py --plan plan_targets.json
```

On macOS you can also run:

```bash
./run.command
```
