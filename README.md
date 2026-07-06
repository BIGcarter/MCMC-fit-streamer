# Modified Mendoza's Infall Model 


## Quick Start

```bash
python run_streamer_model.py      # Single-trajectory visualization
python fit_streamer.py             # MCMC fit (blue lobe, emcee)
python fit_streamer_ns.py          # Nested sampling fit (blue lobe, dynesty, experimental)
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `streamer_ic.py` | Initial condition generators (Mendoza & Ulrich models) |
| `streamer_model.py` | ODE definition + integration (RK45 via solve_ivp) |
| `run_streamer_model.py` | Config → IC → integrate → visualize |
| `fit_streamer.py` | MCMC fitting: Chamfer loss + emcee + Plotly visualization |
| `fit_streamer_red.py` | Same as above, for the red lobe |
| `fit_streamer_ns.py` | Nested sampling: Chamfer loss + dynesty + Plotly |
| `estimate_ic.ipynb` | Jupyter notebook for interactively exploring IC parameters |

## Model Parameters

Modified Mendoza model: radial infall + rigid-body rotation (ω × r) around a user-defined axis, with optional linear drag.

| Parameter | Symbol | Unit | Description | Prior (blue) |
|-----------|--------|------|-------------------|-------------|
| `z` | — | AU | LOS height of the infall starting point |
| `v_r` | — | km/s | Radial velocity at the starting point (infall if negative)|
| `log_omega` | log₁₀(ω) | log₁₀(round/yr) | Rotation angular velocity around the disk axis |
| `theta_axis` | θ_axis | deg | Zenith angle of the rotation axis (from +y toward X-Z plane)|
| `phi_axis` | φ_axis | deg | Azimuth angle of the rotation axis (from +z in X-Z plane)|
| `M` | — | M☉ | Mass of the central protostar|
| `alpha` | α | — | Linear drag coefficient (set large to turn off drag)|
| `x` | — | AU | Sky-plane X-coordinate of the starting point (WCS convention)|
| `y` | — | AU | Sky-plane Y-coordinate of the starting point (WCS convention)|

Fitting scripts use a declarative `PARAM_CONFIG` dict — set `is_constant=True/False` to switch any parameter between free and fixed.

## Coordinate Convention

- **θ** (zenith): measured from **+y** axis toward the X-Z plane (0°–180°)
- **φ** (azimuth): measured from **+z** axis in the X-Z plane, counterclockwise (0°–360°)

## Dependencies

- `numpy`, `scipy` — ODE integration
- `emcee` — MCMC ensemble sampler
- `dynesty` — Dynamic nested sampling
- `corner` — Corner plots
- `plotly` — 3D interactive visualization
- `matplotlib` — Colormaps
- `astropy` — Physical constants
