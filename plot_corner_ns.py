"""
Plot polished corner plot from dynesty nested sampling results.
Edit the configuration section below, then run: python plot_corner_ns.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner

mpl.rcParams['font.family'] = 'serif'


# ============================================================
# Configuration
# ============================================================

SAMPLES_FILE = 'ns_samples_no_pressure_M_26_z_100_south.npz'   # .npz from fit_streamer_ns.py
SAVE_SUFFIX = '_no_pressure_M_26_z_100_south'                   # output file suffix

SMOOTH_1D = True
SHOW_BEST_FIT = True
BINS = 30
COLOR = 'steelblue'        # north: steelblue  red: lightsalmon
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 15
DPI = 150

OUTPUT = None              # None = auto-name: corner_ns_<suffix>.png

PLOT_RANGE = None          # None = auto; or list of [(lo,hi), ...] per param
# PLOT_RANGE = [
#     (80,220),
#     (-2.6,-1.8),
#     (-4.4,-3.5),
#     (18,100),
#     (90,170)
# ]

# ============================================================
# Parameter labels
# ============================================================

# Must match the order of free parameters in fit_streamer_ns.py PARAM_CONFIG.
# Current free params: z, v_r, omega, theta_axis, phi_axis
# Note: omega is linear in the samples; convert to log10 for display.

PARAM_LABELS = [
    r'z [AU]',
    r'v_r [km/s]',
    r'log10(omega) [/yr]',
    r'theta [deg]',
    r'phi [deg]',
]

# ============================================================
# Load & prepare samples
# ============================================================

data = np.load(SAMPLES_FILE)
equal_samples = data['equal_samples']   # (N, ndim) equal-weight posterior
ndim = equal_samples.shape[1]

# Convert omega from linear to log10 for visualization
omega_idx = 2  # 3rd free param (0:z, 1:v_r, 2:omega, 3:theta_axis, 4:phi_axis)
equal_samples[:, omega_idx] = np.log10(equal_samples[:, omega_idx])

q = np.percentile(equal_samples, [16, 50, 84], axis=0)
print(f'Samples: {equal_samples.shape[0]}')
print(f'Median params: {dict(zip([l.strip("$") for l in PARAM_LABELS], q[1]))}')

# MAP (maximum a posteriori) = sample with max log-likelihood
logl = data['logl']
all_samples = data['samples']
map_idx = np.argmax(logl)
map_params = all_samples[map_idx].copy()
map_params[omega_idx] = np.log10(map_params[omega_idx])
print(f'MAP params (max logL = {logl[map_idx]:.2f}): {dict(zip([l.strip("$") for l in PARAM_LABELS], map_params))}\n')

# ============================================================
# Corner plot
# ============================================================

figsize = (min(ndim * 2.5, 14), min(ndim * 2.5, 14))

fig = corner.corner(
    equal_samples,
    labels=PARAM_LABELS,
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
    truths=q[1] if SHOW_BEST_FIT else None,
    truth_color=COLOR,
    range=PLOT_RANGE,
    plot_datapoints=False
)

outfile = OUTPUT
if outfile is None:
    outfile = f'corner_ns{SAVE_SUFFIX}.png'

fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
plt.close()
print(f'Corner plot saved to {outfile}')
