"""
Check MCMC progress mid-run by reading the HDF5 backend file.

Usage (while fit_streamer.py is running):
    python check_mcmc_progress.py [backend_file]

Default: mcmc_chain_blue.h5
"""

import sys
import numpy as np
from emcee.backends import HDFBackend
import matplotlib.pyplot as plt
import corner

backend_file = sys.argv[1] if len(sys.argv) > 1 else 'mcmc_chain_blue.h5'

reader = HDFBackend(backend_file, read_only=True)

try:
    chain = reader.get_chain()
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)

n_steps, n_walkers, ndim = chain.shape
print(f'File: {backend_file}')
print(f'Steps completed: {n_steps}')
print(f'Walkers: {n_walkers}, Dims: {ndim}')
print(f'Production: {n_steps} / ? steps')

# Acceptance fraction from the last few steps
try:
    acc = reader.accepted
    recent_acc = acc[-100:].mean(axis=0)
    print(f'Recent acceptance fraction (last 100 steps): mean={recent_acc.mean():.3f}')
except Exception:
    print('Acceptance fraction not available for partial read')

# Discard first 1/3 as burn-in guess
discard = max(n_steps // 3, 1)
try:
    flat = reader.get_chain(flat=True, discard=discard, thin=max(n_steps // 2000, 1))
    print(f'Flat samples (discard={discard}): {flat.shape[0]}')

    # Quick corner-style text summary
    q = np.percentile(flat, [16, 50, 84], axis=0)
    print('\nParameter ranges (16%, 50%, 84%):')
    for i in range(ndim):
        print(f'  param[{i}]: {q[1,i]:.4g}  (-{q[1,i]-q[0,i]:.3g} / +{q[2,i]-q[1,i]:.3g})')

    # Trace plot
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        for w in range(min(n_walkers, 20)):
            axes[i].plot(chain[:, w, i], 'k', alpha=0.1, linewidth=0.5)
        axes[i].set_ylabel(f'param[{i}]')
        axes[i].axvline(discard, color='r', linestyle='--', alpha=0.5, label=f'discard={discard}')
    axes[-1].set_xlabel('Step')
    axes[0].set_title(f'Trace plots — {backend_file} ({n_steps} steps)')
    fig.tight_layout()
    fig.savefig('mcmc_progress_trace.png', dpi=120)
    print('Trace plot saved to mcmc_progress_trace.png')

    # Corner plot
    labels = [f'param[{i}]' for i in range(ndim)]
    fig_corner = corner.corner(flat, labels=labels, quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs={'fontsize': 10},
                               range = [(-500,500), (-7,-1),(-4.5,-2.5),(0,120),(80,180)] )
    fig_corner.savefig('mcmc_progress_corner.png', dpi=150)
    print('Corner plot saved to mcmc_progress_corner.png')

except Exception as e:
    print(f'Could not compute flat chain: {e}')
