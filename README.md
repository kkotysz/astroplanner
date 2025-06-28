# Astronomical Observation Planner

A simple GUI application for planning astronomical observations. It allows you to:

- Load celestial targets from a JSON or CSV/TSV file.
- Select an observation site and date.
- Calculate visibility curves for each target.
- Plot the results using Matplotlib.
- Display current altitudes, azimuths, sidereal time, and separations from the Moon.

## Features

- Real-time updating of local, UTC, and sidereal times.
- Configurable observation site (latitude, longitude, elevation).
- Automatic calculation of sunrise/sunset, twilight, and moon events.
- Highlight targets above a user-defined altitude limit.
- Dark/light mode toggle.

## Installation

Clone the repository and change into its directory:

```bash
git clone https://github.com/yourusername/astroplanner.git
cd astroplanner
```

### Conda (recommended)

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
