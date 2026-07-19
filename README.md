# Modified Mendoza Infall Model

Ballistic streamer/infall modeling for the protostellar source IRAS 09018-4816. The code integrates trajectories under stellar gravity with optional linear drag and fits them to observed position-position-velocity (PPV) points using MCMC or dynamic nested sampling.

This is a research codebase. Fitting scripts are configured through top-level constants; inspect all priors, fixed parameters, data paths, output suffixes, and sampler settings before starting an expensive run.

## Quick Start

```bash
python run_streamer_model.py          # Integrate and visualize one trajectory
python fit_streamer_ns_codex_v2.py    # Current single-lobe dynamic nested sampling
python fit_streamer_ns_combine.py     # Joint blue/red dynamic nested sampling
```

The nested-sampling configurations use many live points and can run for a long time. For development, prefer `python -m py_compile`, imports, and one short trajectory or likelihood evaluation. Do not start a production sampling run as a smoke test.

## Repository Overview

| Path | Purpose |
|------|---------|
| `streamer_ic.py` | Mendoza/Ulrich initial conditions and coordinate conversion |
| `streamer_model.py` | Gravity/drag ODE, RK45 integration, stopping event, azimuth cutoff, tube sampling |
| `run_streamer_model.py` | Single-trajectory configuration and Plotly visualization |
| `fit_streamer_ns_codex_v2.py` | Current single-lobe dynamic nested sampling with optional flux-weighted PPV loss |
| `fit_streamer_ns_combine.py` | Joint blue/red dynamic nested sampling with per-lobe flux weights |
| `visual_orbit.py` | Single- and combined-lobe PPP/PPV visualization |
| `cluster_ns_samples.py` | Nested-sampling posterior clustering |
| `calc-j.py`, `calc-j-ns.py`, `compare-j.py` | Angular-momentum post-processing |
| `plot_corner.py`, `plot_corner_ns.py` | Posterior plotting utilities |
| `estimate_ic.ipynb`, `channel-peak.ipynb` | Initial-condition and PPV-data exploration |
| `legacy/` | Earlier MCMC and nested-sampling scripts retained for reproducibility |
| `docs/superpowers/` | Historical specifications and implementation plans |

Generated `*.h5`, `*.npz`, `*.png`, and `*.html` files are experiment outputs, not source files. Use a distinct `SAVE_SUFFIX` for every configuration to avoid overwriting previous results.

## Model Parameters

The modified Mendoza model combines radial infall with rigid-body rotation (`omega × r`) around a user-defined axis and optional linear drag.

| Parameter | Unit | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | AU | Cartesian starting position; `z` is the line-of-sight coordinate |
| `v_r` | km/s | Initial radial velocity; negative values describe infall |
| `omega` | round/yr | Rotation angular speed; nested-sampling priors may be uniform in `log10(omega)` |
| `theta_axis` | deg | Rotation-axis angle measured from `+y` |
| `phi_axis` | deg | Rotation-axis angle measured from `+z` in the X-Z plane |
| `M` | solar masses | Central stellar mass |
| `alpha` | integration-time unit | Linear-drag damping timescale; larger values mean weaker drag |

Fitting scripts use `PARAM_CONFIG`. Set `is_constant=True` with a `value` for fixed parameters, or `is_constant=False` with `prior_range=[lo, hi]` for free parameters. An entry with `log_uniform=True` interprets its prior bounds in log10 space and passes `10**value` to the trajectory model.

## Coordinate Convention

The angular convention is not standard spherical coordinates:

- `theta` is measured from `+y` toward the X-Z plane.
- `phi` is measured counterclockwise from `+z` in the X-Z plane.

```text
x = -r * sin(theta) * sin(phi)
y =  r * cos(theta)
z =  r * sin(theta) * cos(phi)
```

Use `cartesian_to_spherical()` in `streamer_ic.py` for the inverse conversion. The integrated `vz` component is used as `v_los`.

## Observational PPV Data

The default observational products are located in the parent directory:

```text
../blue-ppvf.npz
../red-ppvf.npz
```

Each file contains equal-length arrays named `x`, `y`, `v`, and `flux`. The points are channel-by-channel dendrogram-leaf centroids, while `flux` is the corresponding integrated leaf flux.

PPV distances are normalized by default as:

```text
x / SIGMA_XY, y / SIGMA_XY, v / SIGMA_V
SIGMA_XY = 60 AU
SIGMA_V  = 1.331 km/s
```

## Flux-Weighted PPV Loss

The current nested-sampling scripts can weight the observed data-to-model Chamfer term by flux:

```text
loss_dm = sum_i(w_i * d_i**2)
log_likelihood = -0.5 * loss
```

The configuration is:

```python
USE_FLUX_WEIGHTS = True
FLUX_WEIGHT_GAMMA = 1.0
FLUX_CLIP_PERCENTILES = (5.0, 95.0)
```

For each dataset, flux is percentile-clipped, raised to `FLUX_WEIGHT_GAMMA`, and normalized so that the resulting weights have mean one:

```text
w_i = clipped_flux_i**gamma / mean(clipped_flux**gamma)
```

- `USE_FLUX_WEIGHTS=False` disables flux weighting.
- `FLUX_WEIGHT_GAMMA=0` also restores exact equal weighting.
- `gamma=0.5` gives tempered square-root weighting.
- `gamma=1` gives linear flux weighting.

Flux is used here as a geometric importance/reliability weight. The ballistic model does not predict density, excitation, abundance, or radiative transfer, so this must not be interpreted as a physical emission-flux likelihood.

In the combined fit, blue and red weights are normalized independently within each lobe. The joint loss remains the sum over all blue and red points, so the different numbers of observed points still affect the relative lobe contributions.

`fit_streamer_ns_combine.py` also exposes the model-to-data Chamfer term:

```python
USE_MODEL_TO_DATA_LOSS = False
MODEL_TO_DATA_WEIGHT = 1.0
MODEL_TO_DATA_N_SAMPLE = 200
```

The default matches the current v2 single-lobe likelihood and uses only data-to-model distances. Set `USE_MODEL_TO_DATA_LOSS=True` to restore the earlier symmetric Chamfer loss.

## Dynamic Nested Sampling

Both current fitting scripts use `dynesty.DynamicNestedSampler` with multiprocessing, deterministic sampler/resampling seeds, periodic checkpoints, and reproducible equal-weight posterior resampling.

Important settings include:

```text
NLIVE_INIT, NLIVE_BATCH, N_EFFECTIVE, DLOGZ_INIT
N_CPUS, SAMPLER_SEED, RESAMPLE_SEED, CHECKPOINT_EVERY
OBS_DATA or OBS_DATA_BLUE/OBS_DATA_RED
SAVE_SUFFIX
STOPPING_R, AZIMUTH_MAX_DELTA_DEG, T_SPAN, T_EVAL
```

The sample archive records raw/equal-weight samples, `logvol`, `logl`, `logwt`, importance weights, evidence estimates, seeds, and flux/loss configuration. Combined archives additionally save the realized blue and red data weights.

Typical outputs are:

```text
ns_checkpoint<SAVE_SUFFIX>.h5
ns_samples<SAVE_SUFFIX>.npz
ns_summary<SAVE_SUFFIX>.png
ns_trace<SAVE_SUFFIX>.png
ns_corner<SAVE_SUFFIX>.png
ns_bestfit<SAVE_SUFFIX>.html
ns_trajectory<SAVE_SUFFIX>.npz
```

## Validation

Cheap checks should precede any sampling run:

```bash
python -m py_compile streamer_ic.py streamer_model.py
python -m py_compile fit_streamer_ns_codex_v2.py
python -m py_compile fit_streamer_ns_combine.py
```

When changing flux weighting, verify that weights are finite, positive, have mean one, and become all ones with `USE_FLUX_WEIGHTS=False` or `FLUX_WEIGHT_GAMMA=0`.

## Dependencies

- `numpy`, `scipy` — numerical arrays, KDTree queries, and ODE integration
- `astropy` — physical constants and unit conversion
- `dynesty` — dynamic nested sampling
- `emcee` — legacy MCMC workflows
- `matplotlib`, `corner` — diagnostic and posterior plots
- `plotly` — interactive PPP/PPV visualization
