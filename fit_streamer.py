import multiprocessing

import numpy as np
from scipy.spatial import KDTree
import emcee
import corner
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
from matplotlib.cm import RdBu_r

from streamer_ic import (get_streamer_initial_state, cartesian_to_spherical,
                         get_axis_unit_vector, build_local_frame)
from streamer_model import gm_from_mstar, integrate_trajectory, linear_drag, stopping_sphere

# ============================================================
# Parameter configuration (set is_constant=True/False)
# ============================================================

PARAM_CONFIG = {
    'z':           {'is_constant': False, 'prior_range': [50, 1500],     'init': 200,   'label': 'z [AU]'},
    'v_r':         {'is_constant': False, 'prior_range': [-10, 1],       'init': -2,    'label': 'v_r [km/s]'},
    'log_omega':   {'is_constant': False, 'prior_range': [-6, -3],       'init': -4.3,  'label': 'log10(ω) [round/yr]'},
    'theta_axis':  {'is_constant': False, 'prior_range': [0, 90],        'init': 45,    'label': 'θ_axis [deg]'},
    'phi_axis':    {'is_constant': False, 'prior_range': [0, 180],       'init': 120,   'label': 'φ_axis [deg]'},
    'M':           {'is_constant': True,  'prior_range': None,           'init': 15.0,  'label': 'M [M☉]'},
    'alpha':       {'is_constant': True,  'prior_range': None,           'init': 1e6, 'label': 'α'},
    'x':           {'is_constant': True,  'prior_range': None,           'init': -440.0,'label': 'x [AU]'},
    'y':           {'is_constant': True,  'prior_range': None,           'init': 1200.0,'label': 'y [AU]'},
}

# --- Parse PARAM_CONFIG into free / constant ---
_free_names = []
_constant_values = {}
_prior_bounds = []
_init_defaults = []

for _name, _cfg in PARAM_CONFIG.items():
    if _cfg['is_constant']:
        _constant_values[_name] = _cfg['init']
    else:
        _free_names.append(_name)
        _prior_bounds.append(_cfg['prior_range'])
        _init_defaults.append(_cfg['init'])

N_FREE = len(_free_names)
_prior_bounds = np.array(_prior_bounds)
_init_defaults = np.array(_init_defaults)

_param_labels = [PARAM_CONFIG[n]['label'] for n in _free_names]

# --- Remaining fixed config ---
STOPPING_R = 150.0
T_SPAN = (0, 3000)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 1200)

SIGMA_XY = 60.0
SIGMA_V = 1.331

print(f'Free parameters ({N_FREE}): {_free_names}')
print(f'Constants: {list(_constant_values.keys())}')

# ============================================================
# Trajectory wrapper
# ============================================================

def compute_trajectory(params):
    z = params['z']
    x = params.get('x', _constant_values.get('x'))
    y = params.get('y', _constant_values.get('y'))

    r0, theta_part, phi_part = cartesian_to_spherical(x, y, z)

    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=params['v_r'],
        omega_round_yr=10**params['log_omega'],
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)
    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    M_val = params.get('M', _constant_values.get('M'))
    alpha_val = params.get('alpha', _constant_values.get('alpha'))
    GM = gm_from_mstar(M_val)
    drag_func = linear_drag(alpha=alpha_val)

    try:
        sol = integrate_trajectory(
            initial_state, T_SPAN, T_EVAL, GM, drag_func,
            events=stopping_sphere(STOPPING_R)
        )
        if not sol.success:
            return None, None, None, None
        return sol.y[0].copy(), sol.y[1].copy(), sol.y[2].copy(), sol.y[5].copy()
    except Exception:
        return None, None, None, None


# ============================================================
# Chamfer loss in error-normalized PPV space
# ============================================================

def _scale_ppv(traj_ppv):
    scaled = traj_ppv.copy()
    scaled[0] /= SIGMA_XY
    scaled[1] /= SIGMA_XY
    scaled[2] /= SIGMA_V
    return scaled


def chamfer_loss(traj_x, traj_y, traj_v, data_x, data_y, data_v, data_flux,
                 n_sample=200):
    traj_ppv = _scale_ppv(np.array([traj_x, traj_y, traj_v]))
    data_ppv = _scale_ppv(np.array([data_x, data_y, data_v]))

    # data -> model (flux-weighted)
    kd_traj = KDTree(traj_ppv.T)
    d_dm, _ = kd_traj.query(data_ppv.T)
    # loss_dm = np.sum(data_flux * d_dm**2)
    loss_dm = np.sum(d_dm**2)

    # model -> data (unweighted, uniform sample)
    n_pts = traj_ppv.shape[1]
    if n_pts <= n_sample:
        sample_ppv = traj_ppv
    else:
        idx = np.linspace(0, n_pts - 1, n_sample, dtype=int)
        sample_ppv = traj_ppv[:, idx]

    kd_data = KDTree(data_ppv.T)
    d_md, _ = kd_data.query(sample_ppv.T)
    loss_md = np.sum(d_md**2)

    return loss_dm + loss_md
    # return loss_dm 


# ============================================================
# Log-prior & log-probability
# ============================================================

def log_prior(theta):
    for i in range(N_FREE):
        lo, hi = _prior_bounds[i]
        if not (lo <= theta[i] <= hi):
            return -np.inf
    return 0.0


def log_probability(theta, data_x, data_y, data_v, data_flux):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    params = dict(_constant_values)
    for i, name in enumerate(_free_names):
        params[name] = theta[i]

    traj_x, traj_y, traj_z, traj_v = compute_trajectory(params)
    if traj_x is None:
        return -np.inf

    traj_x_range = np.ptp(traj_x)
    traj_y_range = np.ptp(traj_y)
    data_x_range = np.ptp(data_x)
    data_y_range = np.ptp(data_y)
    if traj_x_range < 0.3 * data_x_range and traj_y_range < 0.3 * data_y_range:
        return -np.inf

    loss = chamfer_loss(traj_x, traj_y, traj_v, data_x, data_y, data_v, data_flux)
    return -0.5 * loss


# ============================================================
# MCMC runner
# ============================================================

def run_mcmc(data_x, data_y, data_v, data_flux,
             n_walkers=32, n_burnin=500, n_production=2000,
             init_params=None, rng_seed=42, n_workers=1):
    if init_params is None:
        init_params = _init_defaults.copy()

    ndim = N_FREE
    rng = np.random.default_rng(rng_seed)

    p0 = np.zeros((n_walkers, ndim))
    scatter = np.array([(hi - lo) * 0.05 for lo, hi in _prior_bounds])
    for i in range(n_walkers):
        p0[i] = init_params + scatter * rng.normal(size=ndim)

    # Wrap phi_axis angle if present
    try:
        phi_idx = _free_names.index('phi_axis')
        p0[:, phi_idx] = p0[:, phi_idx] % 360
    except ValueError:
        pass

    moves = emcee.moves.StretchMove(a=1.1)
    pool = None
    if n_workers > 1:
        pool = multiprocessing.Pool(n_workers)
        print(f'Using {n_workers} parallel workers')

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(data_x, data_y, data_v, data_flux),
        moves=moves,
        pool=pool,
    )

    try:
        print(f'Burn-in: {n_burnin} steps...')
        state = sampler.run_mcmc(p0, n_burnin, progress=True)
        sampler.reset()

        print(f'Production: {n_production} steps...')
        sampler.run_mcmc(state, n_production, progress=True)

        flat_samples = sampler.get_chain(flat=True)

        try:
            tau = sampler.get_autocorr_time()
            print(f'Autocorr time: {tau}')
        except Exception:
            print('Autocorr time: not converged')

        print(f'Acceptance fraction: {sampler.acceptance_fraction.mean():.3f}')
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return sampler, flat_samples


# ============================================================
# Visualization
# ============================================================

def _theta_to_params(theta):
    params = dict(_constant_values)
    for i, name in enumerate(_free_names):
        params[name] = theta[i]
    return params


def plot_results(sampler, flat_samples, data_x, data_y, data_v, _data_flux,
                 save_prefix='mcmc_blue', multiple_lines=False, n_lines=100):
    # --- Corner plot ---
    fig_corner = corner.corner(
        flat_samples,
        labels=_param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={'fontsize': 10},
    )
    fig_corner.savefig(f'{save_prefix}_corner.png', dpi=150)
    print(f'Corner plot saved to {save_prefix}_corner.png')

    # --- Best-fit = posterior median ---
    q = np.percentile(flat_samples, [16, 50, 84], axis=0)
    best_theta = q[1]
    best_params = _theta_to_params(best_theta)

    print(f'\nPosterior median (16%, 50%, 84%):')
    for i, label in enumerate(_param_labels):
        val_50 = q[1, i]
        val_16 = q[0, i]
        val_84 = q[2, i]
        print(f'  {label}: {val_50:.3g}  (-{val_50-val_16:.3g} / +{val_84-val_50:.3g})')

    traj_x, traj_y, traj_z, traj_v = compute_trajectory(best_params)

    if traj_x is None:
        print('WARNING: Best-fit trajectory computation failed!')
        return

    # --- Multiple posterior samples ---
    multi_trajs = []  # list of (x, y, z, vlos)
    if multiple_lines and flat_samples.shape[0] > n_lines:
        rng = np.random.default_rng(100)
        idx_samples = rng.choice(flat_samples.shape[0], size=n_lines, replace=False)
        print(f'Computing {n_lines} posterior trajectory samples...')
        for idx in idx_samples:
            p = _theta_to_params(flat_samples[idx])
            tx, ty, tz, tv = compute_trajectory(p)
            if tx is not None:
                multi_trajs.append((tx, ty, tz, tv))
        print(f'  {len(multi_trajs)}/{n_lines} trajectories computed')

    v_range = [-6, 6]
    v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])
    _traj_rgba = RdBu_r(v_norm(traj_v))
    traj_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(
        c[0]*255, c[1]*255, c[2]*255) for c in _traj_rgba]
    _data_rgba = RdBu_r(v_norm(data_v))
    data_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(
        c[0]*255, c[1]*255, c[2]*255) for c in _data_rgba]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('PPP (best fit)', 'PPV (best fit)'),
        horizontal_spacing=0.08,
    )

    # Left: PPP — XYZ position space, colored by V_los
    # Posterior sample trajectories (thin, transparent)
    for tx, ty, tz, tv in multi_trajs:
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode='lines',
            line=dict(color='gray', width=0.5), opacity=0.5,
            showlegend=False,
        ), row=1, col=1)

    # Best-fit trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode='lines+markers', marker=dict(size=3, color=traj_colors, opacity=0.8),
        line=dict(color='gray', width=2), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'), showlegend=False,
    ), row=1, col=1)

    # Rotation axis and equatorial plane (left panel)
    r0_best, _, _ = cartesian_to_spherical(best_params['x'], best_params['y'], best_params['z'])
    n_axis = get_axis_unit_vector(best_params['theta_axis'], best_params['phi_axis'])
    R_plane = build_local_frame(n_axis)
    plane_r = 0.5 * r0_best
    n_ring = 80
    theta_ring = np.linspace(0, 2 * np.pi, n_ring)
    ring_local = np.column_stack([
        plane_r * np.cos(theta_ring),
        plane_r * np.sin(theta_ring),
        np.zeros(n_ring),
    ])
    ring_global = (R_plane @ ring_local.T).T

    tri_i, tri_j, tri_k = [], [], []
    for i in range(n_ring - 1):
        tri_i.append(0); tri_j.append(i + 1); tri_k.append(i + 2)
    tri_i.append(0); tri_j.append(n_ring); tri_k.append(1)

    disc_vertices = np.vstack([[0, 0, 0], ring_global])
    fig.add_trace(go.Mesh3d(
        x=disc_vertices[:, 0], y=disc_vertices[:, 1], z=disc_vertices[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        color='lightblue', opacity=0.25, showlegend=False,
    ), row=1, col=1)

    axis_len = 0.3 * r0_best
    axis_tip = n_axis * axis_len
    fig.add_trace(go.Scatter3d(
        x=[0, axis_tip[0]], y=[0, axis_tip[1]], z=[0, axis_tip[2]],
        mode='lines', line=dict(color='green', width=4), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Cone(
        x=[axis_tip[0]], y=[axis_tip[1]], z=[axis_tip[2]],
        u=[n_axis[0]], v=[n_axis[1]], w=[n_axis[2]],
        sizemode='absolute', sizeref=axis_len * 0.15,
        colorscale=[[0, 'green'], [1, 'green']],
        showscale=False, anchor='tip',
    ), row=1, col=1)

    # Right: PPV
    # Posterior sample trajectories (thin, transparent)
    for tx, ty, tz, tv in multi_trajs:
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tv,
            mode='lines',
            line=dict(color='gray', width=0.5), opacity=0.5,
            showlegend=False,
        ), row=1, col=2)

    # Best-fit trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_v,
        mode='lines+markers', marker=dict(size=3, color=traj_colors, opacity=0.8),
        line=dict(color='gray', width=2), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter3d(
        x=data_x, y=data_y, z=data_v,
        mode='markers',
        marker=dict(size=5, color=data_colors, symbol='circle'),
        error_x=dict(type='data', array=np.full_like(data_x, SIGMA_XY),
                     visible=True, color='gray', width=2),
        error_y=dict(type='data', array=np.full_like(data_y, SIGMA_XY),
                     visible=True, color='gray', width=2),
        error_z=dict(type='data', array=np.full_like(data_v, SIGMA_V),
                     visible=True, color='gray', width=2),
        showlegend=False,
    ), row=1, col=2)

    # Compute PPV aspect ratio to match run_streamer_model.py style
    all_x = np.concatenate([traj_x, data_x])
    all_y = np.concatenate([traj_y, data_y])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_spatial_range = max(x_range, y_range)
    x_scale = x_range / max_spatial_range
    y_scale = y_range / max_spatial_range
    z_scale = 2

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=-2.5),
    )
    fig.update_layout(
        template='plotly_white', showlegend=False,
        scene1=dict(xaxis=dict(title='X', autorange='reversed'), yaxis=dict(title='Y'),
                     zaxis=dict(title='Z'), aspectmode='data'),
        scene2=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='V_los', range=[-20, 20]),
            aspectmode='manual',
            aspectratio=dict(x=x_scale, y=y_scale, z=z_scale),
        ),
        scene_camera=camera,
    )
    fig.write_html(f'{save_prefix}_bestfit.html')
    print(f'Best-fit trajectory saved to {save_prefix}_bestfit.html')

    np.savez(f'{save_prefix}_trajectory.npz',
             x=traj_x, y=traj_y, z=traj_z, v_los=traj_v,
             params=q,
    )
    print(f'Trajectory data saved to {save_prefix}_trajectory.npz')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    data = np.load('../blue-ppvf.npz')
    x_data = data['x']
    y_data = data['y']
    v_data = data['v']
    f_data = data['flux']

    # print(f'  x:    [{x_data.min():.1f}, {x_data.max():.1f}] AU')
    # print(f'  y:    [{y_data.min():.1f}, {y_data.max():.1f}] AU')
    # print(f'  v:    [{v_data.min():.2f}, {v_data.max():.2f}] km/s')
    # print(f'  flux: [{f_data.min():.4f}, {f_data.max():.4f}]')
    # print(f'  sigma_xy = {SIGMA_XY} AU, sigma_v = {SIGMA_V} km/s')
    # print()

    sampler, flat_samples = run_mcmc(x_data, y_data, v_data, f_data,
                                     n_walkers=64, n_burnin=10000, n_production=200000,
                                     n_workers=10)

    np.savez('mcmc_samples_blue.npz',
             chain=sampler.get_chain(),
             flat_samples=flat_samples,
             log_prob=sampler.get_log_prob(),
             acceptance_fraction=sampler.acceptance_fraction,
    )

    plot_results(sampler, flat_samples, x_data, y_data, v_data, f_data, multiple_lines=True,save_prefix="north_streamer_no_pressure")
