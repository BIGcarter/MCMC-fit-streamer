import sys
import numpy as np
from scipy.spatial import KDTree
import multiprocessing as mp
import copy

import dynesty
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
from matplotlib.cm import RdBu_r

from streamer_ic import (get_streamer_initial_state, cartesian_to_spherical,
                         get_axis_unit_vector, build_local_frame)
from streamer_model import gm_from_mstar, integrate_trajectory, linear_drag, stopping_sphere, azimuth_cutoff_idx

# ============================================================
# Configuration
# ============================================================

PARAM_CONFIG = {
    'z':           {'is_constant': False, 'prior_range': [-1500, 200],       'label': r'$z$ [AU]'},  # 0, 1500 north
    'v_r':         {'is_constant': False, 'prior_range': [-10, 1],        'label': r'$v_r$ [km/s]'},
    'omega':       {'is_constant': False, 'prior_range': [-6, -3], 'log_uniform': True, 'label': r'$\log_{10}(\omega)$ [round/yr]'},
    'theta_axis':  {'is_constant': False, 'prior_range': [10, 170],         'label': r'$\theta_{axis}$ [deg]'},
    'phi_axis':    {'is_constant': False, 'prior_range': [10, 350],        'label': r'$\phi_{axis}$ [deg]'},
    'M':           {'is_constant': True,  'value': 10.0,                  'label': r'$M$ [$M_\odot$]'},
    'alpha':       {'is_constant': True,  'value': 1e7,                 'label': r'$\alpha$'},
    'x':           {'is_constant': True,  'value': -440.0,                'label': r'$x$ [AU]'},
    'y':           {'is_constant': True,  'value': -1000,                'label': r'$y$ [AU]'},   # north -500, 1300
}

NLIVE_INIT = 5000
N_CPUS = 10

T_SPAN = (0, 3000)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 1200)
STOPPING_R = 150.0
AZIMUTH_MAX_DELTA_DEG = 200.0

OBS_DATA = '../red-ppvf.npz'
SAVE_SUFFIX = '_no_pressure_south'
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
# Static data & KDTree (built once)
# ============================================================

_data = np.load(OBS_DATA)
_data_x = _data['x']
_data_y = _data['y']
_data_v = _data['v']
_data_flux = _data['flux']

_data_ppv = np.column_stack([
    _data_x / SIGMA_XY,
    _data_y / SIGMA_XY,
    _data_v / SIGMA_V,
])
DATA_KD_TREE = KDTree(_data_ppv)
DATA_XY_XRANGE = np.ptp(_data_x)
DATA_XY_YRANGE = np.ptp(_data_y)

# ============================================================
# Trajectory wrapper
# ============================================================

def _theta_to_params(theta):
    params = dict(_constant_values)
    for i, name in enumerate(_free_names):
        params[name] = theta[i]
    return params


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
        omega_round_yr=params['omega'],
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)
    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    M_val = params.get('M', _constant_values.get('M'))
    alpha_val = params.get('alpha', _constant_values.get('alpha'))
    GM = gm_from_mstar(M_val)
    drag_func = linear_drag(alpha=alpha_val)

    # 初始总能量检查（仅保留束缚轨道）
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

def chamfer_loss(traj_x, traj_y, traj_v, n_sample=200):
    traj_ppv = np.column_stack([
        traj_x / SIGMA_XY, traj_y / SIGMA_XY, traj_v / SIGMA_V,
    ])

    # data -> model
    kd_traj = KDTree(traj_ppv)
    d_dm, _ = kd_traj.query(DATA_KD_TREE.data)
    loss_dm = np.sum(d_dm ** 2)

    # model -> data
    n_pts = traj_ppv.shape[0]
    if n_pts <= n_sample:
        sample_ppv = traj_ppv
    else:
        idx = np.linspace(0, n_pts - 1, n_sample, dtype=int)
        sample_ppv = traj_ppv[idx]

    d_md, _ = DATA_KD_TREE.query(sample_ppv)
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
# Log-likelihood
# ============================================================

def log_likelihood(theta):
    params = _theta_to_params(theta)

    try:
        traj_x, traj_y, traj_z, traj_v = compute_trajectory(params)
    except Exception:
        return -np.inf

    if traj_x is None:
        return -np.inf

    x_range = np.ptp(traj_x)
    y_range = np.ptp(traj_y)
    if x_range < 0.3 * DATA_XY_XRANGE and y_range < 0.3 * DATA_XY_YRANGE:
        return -np.inf

    loss = chamfer_loss(traj_x, traj_y, traj_v)
    return -0.5 * loss


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
        dlogz_init=0.02,
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
    omega_idx = _free_names.index('omega')
    results_viz = copy.deepcopy(results)
    results_viz.samples[:, omega_idx] = np.log10(results_viz.samples[:, omega_idx])

    # --- Summary plot (logZ, dlogZ, nlive vs ncall) ---
    fig_summary, axes_summary = dyplot.runplot(results)
    fig_summary.savefig(f'ns_summary{SAVE_SUFFIX}.png', dpi=150)
    # print('Summary plot saved to ns_summary.png')

    # --- Trace plots ---
    param_labels = _param_labels
    fig_trace, axes_trace = dyplot.traceplot(results_viz, labels=param_labels)
    fig_trace.savefig(f'ns_trace{SAVE_SUFFIX}.png', dpi=150)
    # print('Trace plots saved to ns_trace.png')

    # --- Corner plot ---
    fig_corner, axes_corner = dyplot.cornerplot(results_viz, labels=param_labels,
                                                 quantiles=[0.16, 0.5, 0.84],
                                                 show_titles=True,
                                                 title_kwargs={'fontsize': 10})
    fig_corner.savefig(f'ns_corner{SAVE_SUFFIX}.png', dpi=150)
    # print('Corner plot saved to ns_corner.png')

    # --- Bounding distribution plot (pairwise grid) ---
    # try:
    #     import matplotlib.pyplot as plt
    #     nd = N_FREE
    #     fig_bound, axes = plt.subplots(nd, nd, figsize=(3 * nd, 3 * nd))
    #     for i in range(nd):
    #         for j in range(nd):
    #             ax = axes[i, j] if nd > 1 else axes
    #             if i == j:
    #                 ax.text(0.5, 0.5, _free_names[i], ha='center', va='center',
    #                         transform=ax.transAxes, fontsize=10)
    #                 ax.set_xticks([]); ax.set_yticks([])
    #             elif j < i:
    #                 dyplot.boundplot(results_viz, dims=(j, i), it=-1, ax=ax,
    #                                 show_titles=False)
    #             else:
    #                 ax.axis('off')
    #     fig_bound.tight_layout()
    #     fig_bound.savefig('ns_bound.png', dpi=150)
    # except Exception:
    #     print("bound plot errors.")

    # --- Best-fit: posterior median (more robust than max-likelihood) ---
    best_theta = np.median(equal_samples, axis=0)
    params_bf = _theta_to_params(best_theta)

    print(f'\nBest-fit params (posterior median):')
    for name in _free_names:
        print(f'  {name} = {params_bf[name]:.4g}')
    print(f'  lnZ = {logz:.3f} +/- {logzerr:.3f}')

    traj_x, traj_y, traj_z, traj_v = compute_trajectory(params_bf)
    if traj_x is None:
        print('WARNING: Best-fit trajectory computation failed!')
        sys.exit(1)

    # --- Posterior trajectory samples ---
    N_LINES = 100
    multi_trajs = []
    rng = np.random.default_rng(42)
    idx_samples = rng.choice(len(equal_samples),
                             size=min(N_LINES, len(equal_samples)),
                             replace=False)
    for idx in idx_samples:
        p = equal_samples[idx]
        pm = _theta_to_params(p)
        tx, ty, tz, tv = compute_trajectory(pm)
        if tx is not None:
            multi_trajs.append((tx, ty, tz, tv))
    print(f'Posterior trajectories: {len(multi_trajs)}/{min(N_LINES, len(equal_samples))} computed')

    # --- PPP + PPV Plotly figure ---
    v_range = [-6, 6]
    v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])

    def _to_rgb(arr):
        rgba = RdBu_r(v_norm(arr))
        return ['rgb({:.0f},{:.0f},{:.0f})'.format(c[0]*255, c[1]*255, c[2]*255)
                for c in rgba]

    traj_colors = _to_rgb(traj_v)
    data_colors = _to_rgb(_data_v)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('PPP (best fit)', 'PPV (best fit)'),
        horizontal_spacing=0.08,
    )

    # PPP: posterior samples
    for tx, ty, tz, tv in multi_trajs:
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode='lines',
            line=dict(color='gray', width=0.5), opacity=0.5, showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z, mode='lines+markers',
        marker=dict(size=3, color=traj_colors, opacity=0.8),
        line=dict(color='gray', width=2), showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'), showlegend=False,
    ), row=1, col=1)

    # --- Z=0 reference plane ---
    all_x = np.concatenate([traj_x, _data_x])
    all_y = np.concatenate([traj_y, _data_y])
    px_min, px_max = all_x.min(), all_x.max()
    py_min, py_max = all_y.min(), all_y.max()
    px = np.array([px_min, px_max, px_max, px_min])
    py = np.array([py_min, py_min, py_max, py_max])
    pz = np.zeros(4)
    tri_i_xy = [0, 0]; tri_j_xy = [1, 3]; tri_k_xy = [2, 2]
    fig.add_trace(go.Mesh3d(
        x=px, y=py, z=pz,
        i=tri_i_xy, j=tri_j_xy, k=tri_k_xy,
        color='gray', opacity=0.15, showlegend=False, name='Z=0',
    ), row=1, col=1)

    # Rotation axis + equatorial plane
    r0_bf, _, _ = cartesian_to_spherical(params_bf['x'], params_bf['y'], params_bf['z'])
    n_axis = get_axis_unit_vector(params_bf['theta_axis'], params_bf['phi_axis'])
    R_plane = build_local_frame(n_axis)
    plane_r = 0.5 * r0_bf
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

    axis_len = 0.3 * r0_bf
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

    # PPV: posterior samples
    for tx, ty, tz, tv in multi_trajs:
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tv, mode='lines',
            line=dict(color='gray', width=0.5), opacity=0.5, showlegend=False,
        ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_v, mode='lines+markers',
        marker=dict(size=3, color=traj_colors, opacity=0.8),
        line=dict(color='gray', width=2), showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=_data_x, y=_data_y, z=_data_v, mode='markers',
        marker=dict(size=5, color=data_colors, symbol='circle'),
        error_x=dict(type='data', array=np.full_like(_data_x, SIGMA_XY),
                     visible=True, color='gray', width=2),
        error_y=dict(type='data', array=np.full_like(_data_y, SIGMA_XY),
                     visible=True, color='gray', width=2),
        error_z=dict(type='data', array=np.full_like(_data_v, SIGMA_V),
                     visible=True, color='gray', width=2),
        showlegend=False,
    ), row=1, col=2)

    # Layout
    all_x = np.concatenate([traj_x, _data_x])
    all_y = np.concatenate([traj_y, _data_y])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_spatial_range = max(x_range, y_range)
    x_scale = x_range / max_spatial_range if max_spatial_range > 0 else 1.0
    y_scale = y_range / max_spatial_range if max_spatial_range > 0 else 1.0

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
            xaxis=dict(title='X'), yaxis=dict(title='Y'),
            zaxis=dict(title='V_los', range=[-20, 20]),
            aspectmode='manual',
            aspectratio=dict(x=x_scale, y=y_scale, z=2),
        ),
        scene_camera=camera,
    )
    fig.write_html(f'ns_bestfit{SAVE_SUFFIX}.html')
    # print('Best-fit trajectory saved to ns_bestfit.html')

    # --- Summary percentiles ---
    q = np.percentile(equal_samples, [16, 50, 84], axis=0)
    print(f'\nParameter ranges (16%, 50%, 84%):')
    for i, name in enumerate(_free_names):
        print(f'  {name}: {q[1, i]:.4g}  (-{q[1, i] - q[0, i]:.3g} / +{q[2, i] - q[1, i]:.3g})')

    np.savez(f'ns_trajectory{SAVE_SUFFIX}.npz',
             x=traj_x, y=traj_y, z=traj_z, v_los=traj_v,
             params=q,
    )
    print(f'Trajectory data saved to ns_trajectory{SAVE_SUFFIX}.npz')
