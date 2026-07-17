import sys
import numpy as np
from scipy.spatial import KDTree
import multiprocessing as mp
import copy

import dynesty
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal

from streamer_ic import get_streamer_initial_state, cartesian_to_spherical
from streamer_model import gm_from_mstar, integrate_trajectory, linear_drag, stopping_sphere, azimuth_cutoff_idx
from visual_orbit import plot_orbit_combined

# ============================================================
# Configuration
# ============================================================
#
# Joint fit of the blue (north) and red (south) lobes.
#   - Shared free parameter:   M
#   - Shared constant:         alpha = 1e7
#   - Per-lobe free params:    z, v_r, omega, theta_axis, phi_axis
#   - Per-lobe constants:      x, y
# Parameter keys are suffixed with the lobe name (_blue / _red).

PARAM_CONFIG = {
    # --- shared ---
    'M':                {'is_constant': True, 'value': 26,       'label': r'$M$ [$M_\odot$]'},
    'alpha':            {'is_constant': True,  'value': 1e7,                 'label': r'$\alpha$'},

    # --- blue (north) ---
    'z_blue':           {'is_constant': False, 'prior_range': [-100, 1500],  'label': r'$z_{\rm b}$ [AU]'},
    'v_r_blue':         {'is_constant': False, 'prior_range': [-10, 1],      'label': r'$v_{r,\rm b}$ [km/s]'},
    'omega_blue':       {'is_constant': False, 'prior_range': [-6, -3], 'log_uniform': True, 'label': r'$\log_{10}(\omega_{\rm b})$ [round/yr]'},
    'theta_axis_blue':  {'is_constant': False, 'prior_range': [0, 90],       'label': r'$\theta_{\rm axis,b}$ [deg]'},
    'phi_axis_blue':    {'is_constant': False, 'prior_range': [0, 180],      'label': r'$\phi_{\rm axis,b}$ [deg]'},
    'x_blue':           {'is_constant': True,  'value': -500.0,              'label': r'$x_{\rm b}$ [AU]'},
    'y_blue':           {'is_constant': True,  'value': 1200.0,              'label': r'$y_{\rm b}$ [AU]'},

    # --- red (south) ---
    'z_red':            {'is_constant': False, 'prior_range': [-3000, -100],  'label': r'$z_{\rm r}$ [AU]'},
    'v_r_red':          {'is_constant': False, 'prior_range': [-10, 1],      'label': r'$v_{r,\rm r}$ [km/s]'},
    'omega_red':        {'is_constant': False, 'prior_range': [-6, -3], 'log_uniform': True, 'label': r'$\log_{10}(\omega_{\rm r})$ [round/yr]'},
    'theta_axis_red':   {'is_constant': False, 'prior_range': [0, 90],       'label': r'$\theta_{\rm axis,r}$ [deg]'},
    'phi_axis_red':     {'is_constant': False, 'prior_range': [0, 180],      'label': r'$\phi_{\rm axis,r}$ [deg]'},
    'x_red':            {'is_constant': True,  'value': -440.0,              'label': r'$x_{\rm r}$ [AU]'},
    'y_red':            {'is_constant': True,  'value': -1000.0,             'label': r'$y_{\rm r}$ [AU]'},
}

# Canonical per-lobe parameter names (suffix stripped) fed to compute_trajectory.
_LOBE_KEYS = ['z', 'v_r', 'omega', 'theta_axis', 'phi_axis', 'x', 'y']
LOBES = ('blue', 'red')

NLIVE_INIT = 5000
N_CPUS = 10

T_SPAN = (0, 3000)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 1200)
STOPPING_R = 150.0
AZIMUTH_MAX_DELTA_DEG = 200.0

OBS_DATA_BLUE = '../blue-ppvf.npz'
OBS_DATA_RED = '../red-ppvf.npz'
SAVE_SUFFIX = '_M_26_combined'
SIGMA_XY = 60.0
SIGMA_V = 1.331

# --- Parse PARAM_CONFIG into free / constant ---
_free_names = []
_constant_values = {}
_prior_bounds = []
_prior_log_uniform = []

for _name, _cfg in PARAM_CONFIG.items():
    if _cfg['is_constant']:
        _constant_values[_name] = _cfg['value']
    else:
        _free_names.append(_name)
        _prior_bounds.append(_cfg['prior_range'])
        _prior_log_uniform.append(_cfg.get('log_uniform', False))

N_FREE = len(_free_names)
_prior_bounds = np.array(_prior_bounds)
_param_labels = [PARAM_CONFIG[n]['label'] for n in _free_names]

print(f'Free parameters ({N_FREE}): {_free_names}')
print(f'Constants: {list(_constant_values.keys())}')

# ============================================================
# Static data & KDTrees (built once, one per lobe)
# ============================================================

def _load_ppv(path):
    d = np.load(path)
    return d['x'], d['y'], d['v'], d['flux']


def _build_kdtree(x, y, v):
    ppv = np.column_stack([x / SIGMA_XY, y / SIGMA_XY, v / SIGMA_V])
    return KDTree(ppv)


_blue_x, _blue_y, _blue_v, _blue_flux = _load_ppv(OBS_DATA_BLUE)
_red_x, _red_y, _red_v, _red_flux = _load_ppv(OBS_DATA_RED)

DATA_KD_TREE = {
    'blue': _build_kdtree(_blue_x, _blue_y, _blue_v),
    'red':  _build_kdtree(_red_x, _red_y, _red_v),
}
DATA_XY_RANGE = {
    'blue': (np.ptp(_blue_x), np.ptp(_blue_y)),
    'red':  (np.ptp(_red_x), np.ptp(_red_y)),
}
DATA_XYV = {
    'blue': (_blue_x, _blue_y, _blue_v),
    'red':  (_red_x, _red_y, _red_v),
}

# ============================================================
# Parameter helpers
# ============================================================

def _theta_to_params(theta):
    params = dict(_constant_values)
    for i, name in enumerate(_free_names):
        params[name] = theta[i]
    return params


def _lobe_params(all_params, lobe):
    """Extract a canonical single-lobe param dict from the full param set.

    Maps lobe-suffixed keys (e.g. ``z_blue``) to canonical names (``z``) and
    injects the shared ``M`` / ``alpha``.
    """
    p = {k: all_params[f'{k}_{lobe}'] for k in _LOBE_KEYS}
    p['M'] = all_params['M']
    p['alpha'] = all_params['alpha']
    return p


# ============================================================
# Trajectory wrapper
# ============================================================

def compute_trajectory(params):
    z = params['z']
    x = params['x']
    y = params['y']

    r0, theta_part, phi_part = cartesian_to_spherical(x, y, z)

    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=params['v_r'],
        omega_round_yr=params['omega'],
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)
    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    GM = gm_from_mstar(params['M'])
    drag_func = linear_drag(alpha=params['alpha'])

    r0 = np.sqrt(x0**2 + y0**2 + z0**2)
    E0 = 0.5 * (vx0**2 + vy0**2 + vz0**2) - GM / r0
    if E0 >= 0:
        return None, None, None, None

    try:
        sol = integrate_trajectory(initial_state, T_SPAN, T_EVAL, GM, drag_func, events=stopping_sphere(STOPPING_R))
        if not sol.success:
            return None, None, None, None
        x_arr = sol.y[0].copy()
        y_arr = sol.y[1].copy()
        z_arr = sol.y[2].copy()
        v_arr = sol.y[5].copy()

        cut = azimuth_cutoff_idx(x_arr, y_arr, max_delta_deg=AZIMUTH_MAX_DELTA_DEG)
        return x_arr[:cut+1], y_arr[:cut+1], z_arr[:cut+1], v_arr[:cut+1]
    except Exception:
        return None, None, None, None


# ============================================================
# Chamfer loss in error-normalized PPV space
# ============================================================

def chamfer_loss(traj_x, traj_y, traj_v, data_kd_tree, n_sample=200):
    traj_ppv = np.column_stack([
        traj_x / SIGMA_XY, traj_y / SIGMA_XY, traj_v / SIGMA_V,
    ])

    # data -> model
    kd_traj = KDTree(traj_ppv)
    d_dm, _ = kd_traj.query(data_kd_tree.data)
    loss_dm = np.sum(d_dm ** 2)

    # model -> data
    n_pts = traj_ppv.shape[0]
    if n_pts <= n_sample:
        sample_ppv = traj_ppv
    else:
        idx = np.linspace(0, n_pts - 1, n_sample, dtype=int)
        sample_ppv = traj_ppv[idx]

    d_md, _ = data_kd_tree.query(sample_ppv)
    loss_md = np.sum(d_md ** 2)

    return loss_dm + loss_md


# ============================================================
# Prior transform (unit cube -> parameter space)
# ============================================================

def prior_transform(u):
    theta = np.empty(N_FREE)
    for i in range(N_FREE):
        lo, hi = _prior_bounds[i]
        val = lo + u[i] * (hi - lo)
        if _prior_log_uniform[i]:
            theta[i] = 10.0 ** val
        else:
            theta[i] = val
    return theta


# ============================================================
# Log-likelihood (joint over both lobes)
# ============================================================

def log_likelihood(theta):
    all_params = _theta_to_params(theta)

    total_loss = 0.0
    for lobe in LOBES:
        p = _lobe_params(all_params, lobe)
        try:
            traj_x, traj_y, traj_z, traj_v = compute_trajectory(p)
        except Exception:
            return -np.inf

        if traj_x is None:
            return -np.inf

        x_range = np.ptp(traj_x)
        y_range = np.ptp(traj_y)
        xr, yr = DATA_XY_RANGE[lobe]
        if x_range < 0.3 * xr and y_range < 0.3 * yr:
            return -np.inf

        total_loss += chamfer_loss(traj_x, traj_y, traj_v, DATA_KD_TREE[lobe])

    return -0.5 * total_loss


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    pool = mp.Pool(N_CPUS) if N_CPUS > 1 else None
    if pool is not None:
        pool.size = N_CPUS

    sampler = DynamicNestedSampler(
        log_likelihood,
        prior_transform,
        ndim=N_FREE,
        nlive=NLIVE_INIT,
        bound='multi',
        pool=pool,
        update_interval=1.5,
        sample='rslice',
        slices=10,
    )

    sampler.run_nested(
        dlogz_init=0.05,
        nlive_batch=5000,
        checkpoint_file=f'ns_checkpoint{SAVE_SUFFIX}.h5',
        )
    print('Done. Wake up.')

    if pool is not None:
        pool.close()
        pool.join()

    # --- Post-processing ---
    results = sampler.results
    samples = results.samples
    logvol = results.logvol
    logl = results.logl
    logz = results.logz[-1]
    logzerr = results.logzerr[-1]

    print(f'log Z = {logz:.3f} +/- {logzerr:.3f}')

    weights = np.exp(logvol + logl - logz)
    equal_samples = resample_equal(samples, weights)

    print(f'Equal-weight posterior samples: {equal_samples.shape[0]}')

    np.savez(f'ns_samples{SAVE_SUFFIX}.npz',
             samples=samples,
             equal_samples=equal_samples,
             logvol=logvol,
             logl=logl,
             logz=logz,
             logzerr=logzerr,
    )

    # --- Build visualization results with omega in log10 space ---
    omega_indices = [i for i, n in enumerate(_free_names) if n.startswith('omega')]
    results_viz = copy.deepcopy(results)
    for oi in omega_indices:
        results_viz.samples[:, oi] = np.log10(results_viz.samples[:, oi])

    # --- Summary plot (logZ, dlogZ, nlive vs ncall) ---
    fig_summary, axes_summary = dyplot.runplot(results)
    fig_summary.savefig(f'ns_summary{SAVE_SUFFIX}.png', dpi=150)

    # --- Trace plots ---
    param_labels = _param_labels
    fig_trace, axes_trace = dyplot.traceplot(results_viz, labels=param_labels)
    fig_trace.savefig(f'ns_trace{SAVE_SUFFIX}.png', dpi=150)

    # --- Corner plot ---
    fig_corner, axes_corner = dyplot.cornerplot(results_viz, labels=param_labels,
                                                 quantiles=[0.16, 0.5, 0.84],
                                                 show_titles=True,
                                                 title_kwargs={'fontsize': 10})
    fig_corner.savefig(f'ns_corner{SAVE_SUFFIX}.png', dpi=150)

    # --- Best-fit: posterior median (more robust than max-likelihood) ---
    best_theta = np.median(equal_samples, axis=0)
    all_params_bf = _theta_to_params(best_theta)

    print(f'\nBest-fit params (posterior median):')
    for name in _free_names:
        print(f'  {name} = {all_params_bf[name]:.4g}')
    print(f'  lnZ = {logz:.3f} +/- {logzerr:.3f}')

    # --- Best-fit trajectory + posterior samples per lobe ---
    N_LINES = 100
    rng = np.random.default_rng(42)
    idx_samples = rng.choice(len(equal_samples),
                             size=min(N_LINES, len(equal_samples)),
                             replace=False)

    lobes_viz = []
    traj_save = {}
    for lobe in LOBES:
        params_bf = _lobe_params(all_params_bf, lobe)
        traj_x, traj_y, traj_z, traj_v = compute_trajectory(params_bf)
        if traj_x is None:
            print(f'WARNING: Best-fit {lobe} trajectory computation failed!')
            sys.exit(1)

        multi_trajs = []
        for idx in idx_samples:
            pm = _lobe_params(_theta_to_params(equal_samples[idx]), lobe)
            tx, ty, tz, tv = compute_trajectory(pm)
            if tx is not None:
                multi_trajs.append((tx, ty, tz, tv))
        print(f'Posterior trajectories ({lobe}): {len(multi_trajs)}/{len(idx_samples)} computed')

        data_x, data_y, data_v = DATA_XYV[lobe]
        lobes_viz.append(dict(
            traj_x=traj_x, traj_y=traj_y, traj_z=traj_z, traj_v=traj_v,
            data_x=data_x, data_y=data_y, data_v=data_v,
            geom_x=params_bf['x'], geom_y=params_bf['y'], geom_z=params_bf['z'],
            theta_axis_deg=params_bf['theta_axis'], phi_axis_deg=params_bf['phi_axis'],
            multi_trajs=multi_trajs,
        ))
        traj_save[f'{lobe}_x'] = traj_x
        traj_save[f'{lobe}_y'] = traj_y
        traj_save[f'{lobe}_z'] = traj_z
        traj_save[f'{lobe}_v_los'] = traj_v

    # --- Combined PPP + PPV Plotly figure (both lobes overlaid) ---
    plot_orbit_combined(
        lobes_viz,
        sigma_xy=SIGMA_XY, sigma_v=SIGMA_V,
        v_range=(-6, 6),
        output_html=f'ns_bestfit{SAVE_SUFFIX}.html',
    )

    # --- Summary percentiles ---
    q = np.percentile(equal_samples, [16, 50, 84], axis=0)
    print(f'\nParameter ranges (16%, 50%, 84%):')
    for i, name in enumerate(_free_names):
        print(f'  {name}: {q[1, i]:.4g}  (-{q[1, i] - q[0, i]:.3g} / +{q[2, i] - q[1, i]:.3g})')

    np.savez(f'ns_trajectory{SAVE_SUFFIX}.npz',
             params=q,
             free_names=np.array(_free_names),
             **traj_save,
    )
    print(f'Trajectory data saved to ns_trajectory{SAVE_SUFFIX}.npz')
