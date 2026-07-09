#!/usr/bin/env python3
"""
Compute specific angular momentum distribution from MCMC chain samples.

For each walker sample, computes the specific angular momentum vector
j = r × v at the initial condition, then derives its magnitude |j|
and direction (θ_j, φ_j) in the model's spherical convention.

Outputs:
  - angular_momentum_samples.npz
  - angular_momentum_corner.png
"""

import numpy as np
from emcee.backends import HDFBackend
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner

from streamer_ic import get_streamer_initial_state, cartesian_to_spherical

mpl.rcParams['font.family'] = 'serif'

# ============================================================
# Configuration
# ============================================================

CHAIN_FILE = 'mcmc_chain_red_bk.h5'  # 'mcmc_chain_blue.h5'
DISCARD = 0            # burn-in steps (0 = auto: first 1/3)
THIN = 1               # thinning factor (1 = auto: target ~3000 samples)

SMOOTH_1D = True
SHOW_BEST_FIT = True
BINS = 30
COLOR = 'lightsalmon'  # north: steelblue  red: lightsalmon
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 15
DPI = 150

OUTPUT_PREFIX = 'angular_momentum_south'

# Constants matching fit_streamer.py PARAM_CONFIG defaults
X_FIXED = -440.0    # -500 north
Y_FIXED = -1100.0   # 1300 north
M_STAR = 10.0       # not used for j (purely kinematic), kept for reference

# Free parameter names (order must match the chain columns)
FREE_NAMES = ['z', 'v_r', 'log_omega', 'theta_axis', 'phi_axis']

# PLOT_RANGE = [
#     (2000,9000),
#     (65,85),
#     (100,170)
# ]

## RED
PLOT_RANGE = [
    (7200, 10000),
    (57, 64),
    (100, 109),
]

# ============================================================
# Load chain
# ============================================================

reader = HDFBackend(CHAIN_FILE, read_only=True)
chain = reader.get_chain()
n_steps, n_walkers, ndim = chain.shape

if DISCARD <= 0:
    discard = n_steps // 3
else:
    discard = DISCARD

if THIN <= 1:
    target = 3000
    thin = max((n_steps - discard) * n_walkers // target, 1)
else:
    thin = THIN

flat_samples = reader.get_chain(flat=True, discard=discard, thin=thin)
log_prob = reader.get_log_prob(flat=True, discard=discard, thin=thin)

bf_idx = np.argmax(log_prob)
bf_sample = flat_samples[bf_idx]

print(f'Loaded {flat_samples.shape[0]} samples from {CHAIN_FILE}')
print(f'  discard={discard}, thin={thin}')
print(f'  Chain shape: {chain.shape}')
print(f'  Parameters: {FREE_NAMES}')

# ============================================================
# Compute specific angular momentum for each sample
# ============================================================

j_mags = []
theta_js = []
phi_js = []

for theta in flat_samples:
    params = dict(zip(FREE_NAMES, theta))
    z = params['z']

    r0, theta_part, phi_part = cartesian_to_spherical(X_FIXED, Y_FIXED, z)

    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=params['v_r'],
        omega_round_yr=10 ** params['log_omega'],
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)

    r_vec = np.array([x0, y0, z0])
    v_vec = np.array([vx0, vy0, vz0])

    j_vec = np.cross(r_vec, v_vec)
    j_mag = np.linalg.norm(j_vec)

    _, theta_j, phi_j = cartesian_to_spherical(j_vec[0], j_vec[1], j_vec[2])

    j_mags.append(j_mag)
    theta_js.append(theta_j)
    phi_js.append(phi_j)

j_mags = np.array(j_mags)
theta_js = np.array(theta_js)
phi_js = np.array(phi_js)


# ============================================================
# Statistics
# ============================================================

samples_3d = np.column_stack([j_mags, theta_js, phi_js])
q = np.percentile(samples_3d, [16, 50, 84], axis=0)

bf_p = dict(zip(FREE_NAMES, bf_sample))
bf_r0, bf_tp, bf_pp = cartesian_to_spherical(X_FIXED, Y_FIXED, bf_p['z'])
bf_x0, bf_y0, bf_z0, bf_vx0, bf_vy0, bf_vz0 = get_streamer_initial_state('mendoza',
    r0=bf_r0, theta_part_deg=bf_tp, phi_part_deg=bf_pp,
    v_r=bf_p['v_r'], omega_round_yr=10 ** bf_p['log_omega'],
    theta_axis_deg=bf_p['theta_axis'], phi_axis_deg=bf_p['phi_axis'])
bf_j_vec = np.cross([bf_x0, bf_y0, bf_z0], [bf_vx0, bf_vy0, bf_vz0])
bf_j_mag = np.linalg.norm(bf_j_vec)
_, bf_theta_j, bf_phi_j = cartesian_to_spherical(bf_j_vec[0], bf_j_vec[1], bf_j_vec[2])
# bf_j = np.array([bf_j_mag, bf_theta_j, bf_phi_j])
bf_j = q[1]

print('\nSpecific angular momentum (max log_prob sample):')
for name, val in [('|j| [AU·km/s]', bf_j[0]),
                   ('θ_j [deg]', bf_j[1]),
                   ('φ_j [deg]', bf_j[2])]:
    print(f'  {name}: {val:.4g}')

print('\nSpecific angular momentum distribution:')
for i, name in enumerate(['|j| [AU·km/s]', 'θ_j [deg]', 'φ_j [deg]']):
    print(f'  {name}: {q[1,i]:.4g}  (-{q[1,i] - q[0,i]:.4g} / +{q[2,i] - q[1,i]:.4g})')

# ============================================================
# Save npz
# ============================================================

np.savez(
    f'{OUTPUT_PREFIX}_samples.npz',
    j_mag=j_mags,
    theta_j=theta_js,
    phi_j=phi_js,
    flat_samples=flat_samples,
    log_prob=log_prob,
    free_names=FREE_NAMES,
    x_fixed=X_FIXED,
    y_fixed=Y_FIXED,
    best_fit=bf_j,
)
print(f'\nSaved to {OUTPUT_PREFIX}_samples.npz')

# ============================================================
# Corner plot
# ============================================================

labels = [
    r'$|j|$ [AU$\cdot$km/s]',
    r'$\theta_j$ [$^\circ$]',
    r'$\phi_j$ [$^\circ$]',
]

ndim_j = 3
figsize = (min(ndim_j * 2.5, 14), min(ndim_j * 2.5, 14))

fig = corner.corner(
    samples_3d,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.2f',
    title_kwargs={'fontsize': TITLE_FONT_SIZE},
    label_kwargs={'fontsize': LABEL_FONT_SIZE, 'weight': 'bold'},
    smooth=SMOOTH_1D,
    smooth1d=SMOOTH_1D,
    bins=BINS,
    color=COLOR,
    hist_kwargs={'linewidth': 5, 'alpha': 0.6},
    fig=plt.figure(figsize=figsize),
    truths=bf_j if SHOW_BEST_FIT else None,
    truth_color=COLOR,
    range=PLOT_RANGE,
)

outfile = f'{OUTPUT_PREFIX}_corner.png'
fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
print(f'Corner plot saved to {outfile}')
plt.close(fig)
