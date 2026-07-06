# Ballistic Infall Model for IRAS 09018-4816

3D ballistic infall model (streamer) for protostellar source IRAS 09018-4816.

## Quick Start

```bash
python run_streamer_model.py      # Single-trajectory visualization
python fit_streamer.py             # MCMC fit (blue lobe, emcee)
python fit_streamer_red.py         # MCMC fit (red lobe, emcee)
python fit_streamer_ns.py          # Nested sampling fit (blue lobe, dynesty)
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

## Parameters

All fitting scripts use a declarative `PARAM_CONFIG` dict. Set `is_constant=True/False` to switch between free and fixed parameters.

### Free parameters (Mendoza model)

| Parameter | Prior range | Description |
|-----------|------------|-------------|
| `z` | [50, 1500] or [−1000, 0] AU | Height of infall starting point |
| `v_r` | [−10, 1] km/s | Radial velocity |
| `log_omega` | [−6, −3] | log₁₀ rotation rate (round/yr) |
| `theta_axis` | [0, 90]° | Rotation axis zenith angle |
| `phi_axis` | [0, 180]° | Rotation axis azimuth angle |

### Fixed parameters

| Parameter | Value (blue) | Value (red) |
|-----------|-------------|-------------|
| M | 15 M☉ | 15 M☉ |
| alpha | 500 (or 1e6) | 500 |
| x | −440 AU | −440 AU |
| y | 1200 AU | −1000 AU |

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
