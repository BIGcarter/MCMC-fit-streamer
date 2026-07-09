"""
Plot polished corner plot from MCMC HDF5 chain.
Edit the configuration section below, then run: python plot_corner.py
"""

import numpy as np
from emcee.backends import HDFBackend
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner

mpl.rcParams['font.family'] = 'serif'


BACKEND_FILE = 'mcmc_chain_red_bk.h5' # 'mcmc_chain_blue.h5'
DISCARD = 0          # burn-in steps (0 = auto: first 1/3)
THIN = 1             # thinning factor (1 = auto: target ~3000 samples)

SMOOTH_1D = True     
SHOW_BEST_FIT = True  
SHOW_PRIOR = False     
BINS = 30            
COLOR = 'lightsalmon'  # north: steelblue  red: lightsalmon
TITLE_FONT_SIZE = 11       
LABEL_FONT_SIZE = 15       
DPI = 150            

OUTPUT = None        


# Blue range
# PLOT_RANGE = [
#             (-75,40),
#             (-2.9,-2.2),
#             (-4.3,-3.2),
#             (20,100),
#             (100,180)
#             ]   

# Red range
PLOT_RANGE = None
# PLOT_RANGE = [
#             (-1200,-450),
#             (-6,-2),
#             (-4,-3.52),
#             (0,100),
#             (0,150)
#             ]  



# ============================================================
# Parameter labels and prior ranges
# ============================================================

PARAM_LABELS = [
    r'$z$ [AU]',
    r'$v_r$ [km s$^{-1}$]',
    r'$\log_{10}(\omega)$ [yr$^{-1}$]',
    r'$\theta_\mathrm{axis}$ [$^\circ$]',
    r'$\phi_\mathrm{axis}$ [$^\circ$]',
]

# PRIOR_RANGES = [
#     (-100, 1500),
#     (-10, 1),
#     (-6, -3),
#     (0, 90),
#     (0, 180),
# ]



reader = HDFBackend(BACKEND_FILE, read_only=True)
chain = reader.get_chain()
n_steps, n_walkers, ndim = chain.shape

if DISCARD <= 0:
    discard = n_steps // 3
else:
    discard = DISCARD

# Thinning, not reading all the steps
if THIN <= 1:
    target = 3000
    thin = max((n_steps - discard) * n_walkers // target, 1)
else:
    thin = THIN

flat = reader.get_chain(flat=True, discard=discard, thin=thin)
log_prob = reader.get_log_prob(flat=True, discard=discard, thin=thin)


bf_idx = np.argmax(log_prob)
bf_sample = flat[bf_idx]
q = np.percentile(flat, [16, 50, 84], axis=0)


# ============================================================
# Corner plot
# ============================================================


figsize = (min(ndim * 2.5, 14), min(ndim * 2.5, 14))

fig = corner.corner(
    flat,
    labels=PARAM_LABELS,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.2f',
    title_kwargs={'fontsize': 11},
    label_kwargs={'fontsize': 15, 'weight': 'bold'},
    smooth=SMOOTH_1D,
    smooth1d=SMOOTH_1D,
    bins=BINS,
    color=COLOR,
    hist_kwargs={'linewidth':5, 'alpha': 0.6},
    fig=plt.figure(figsize=figsize),
    truths=q[1] if SHOW_BEST_FIT else None,
    truth_color=COLOR, # crimson
    range=PLOT_RANGE,
)

outfile = OUTPUT
if outfile is None:
    base = BACKEND_FILE.rsplit('.', 1)[0]
    outfile = f'corner-{base}.png'

fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
plt.close()
