"""Standalone orbit visualization: PPP + PPV Plotly figure.

Extracted from fit_streamer_ns.py. Renders the best-fit streamer trajectory
in position space (PPP: X-Y-Z) and position-velocity space (PPV: X-Y-V_los),
overlaid with observed PPV data and optional posterior-sample trajectories.

Usage:
    - As a library:  from visual_orbit import plot_orbit
    - Standalone:     python visual_orbit.py   (edit the CONFIG section below)
"""
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
from matplotlib.cm import RdBu_r

from streamer_ic import (get_streamer_initial_state, cartesian_to_spherical,
                         get_axis_unit_vector, build_local_frame)
from streamer_model import (gm_from_mstar, integrate_trajectory, linear_drag,
                            stopping_sphere, azimuth_cutoff_idx)


def compute_model_trajectory(
    x, y, z, v_r, log10_omega, theta_axis_deg, phi_axis_deg,
    M=10.0, alpha=500.0,
    t_span=(0, 3000), n_eval=1200,
    stopping_r=150.0, azimuth_max_delta_deg=200.0,
):
    """Integrate a Mendoza streamer trajectory from model parameters.

    Parameters
    ----------
    x, y, z : float
        Starting position [AU].
    v_r : float
        Radial (infall) velocity [km/s].
    log10_omega : float
        log10 of the rotation rate [round/yr]; converted internally via
        ``omega = 10 ** log10_omega``.
    theta_axis_deg, phi_axis_deg : float
        Rotation-axis orientation (streamer_ic convention) [deg].
    M : float
        Central stellar mass [M_sun].
    alpha : float
        Linear-drag damping coefficient.
    t_span : (t0, t1)
        Integration time span [yr].
    n_eval : int
        Number of evaluation points.
    stopping_r : float
        Integration terminates when r crosses this radius [AU].
    azimuth_max_delta_deg : float
        Post-hoc azimuth-cutoff threshold [deg].

    Returns
    -------
    (x, y, z, v_los) arrays, or (None, None, None, None) on failure
    (unbound orbit or solver failure).
    """
    r0, theta_part, phi_part = cartesian_to_spherical(x, y, z)
    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=v_r,
        omega_round_yr=10.0 ** log10_omega,
        theta_axis_deg=theta_axis_deg,
        phi_axis_deg=phi_axis_deg,
    )
    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)
    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    GM = gm_from_mstar(M)
    drag_func = linear_drag(alpha=alpha)

    # Keep only bound orbits (E < 0).
    rr = np.sqrt(x0**2 + y0**2 + z0**2)
    E0 = 0.5 * (vx0**2 + vy0**2 + vz0**2) - GM / rr
    # if E0 >= 0:
    #     print("E>0, unbound.")
    #     return None, None, None, None

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = integrate_trajectory(initial_state, t_span, t_eval, GM, drag_func,
                               events=stopping_sphere(stopping_r))
    if not sol.success:
        return None, None, None, None

    x_arr = sol.y[0]
    y_arr = sol.y[1]
    z_arr = sol.y[2]
    v_arr = sol.y[5]
    cut = azimuth_cutoff_idx(x_arr, y_arr, max_delta_deg=azimuth_max_delta_deg)
    return x_arr[:cut + 1], y_arr[:cut + 1], z_arr[:cut + 1], v_arr[:cut + 1]


def plot_orbit(
    traj_x, traj_y, traj_z, traj_v,
    data_x, data_y, data_v,
    geom_x, geom_y, geom_z, theta_axis_deg, phi_axis_deg,
    multi_trajs=None,
    sigma_xy=60.0,
    sigma_v=1.331,
    v_range=(-6.0, 6.0),
    vlos_axis_range=(-20, 20),
    output_html='orbit.html',
    subplot_titles=('PPP (best fit)', 'PPV (best fit)'),
):
    """Render PPP + PPV Plotly figure and write it to ``output_html``.

    Parameters
    ----------
    traj_x, traj_y, traj_z, traj_v : array
        Best-fit trajectory (position + line-of-sight velocity).
    data_x, data_y, data_v : array
        Observed PPV data points.
    geom_x, geom_y, geom_z : float
        Best-fit starting position, used for the rotation axis / equatorial
        plane geometry (sets characteristic radius r0).
    theta_axis_deg, phi_axis_deg : float
        Best-fit rotation-axis orientation (same convention as streamer_ic).
    multi_trajs : list of (x, y, z, v) tuples, optional
        Posterior-sample trajectories drawn as thin gray lines.
    sigma_xy, sigma_v : float
        Observational uncertainties (error bars on the PPV data points).
    v_range : (lo, hi)
        Color-scale range for V_los (RdBu_r).
    vlos_axis_range : (lo, hi)
        Fixed axis range for the V_los axis in the PPV panel.
    output_html : str
        Output HTML path.
    subplot_titles : (str, str)
        Titles for the PPP and PPV panels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    multi_trajs = multi_trajs or []

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    data_v = np.asarray(data_v)

    v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])

    def _to_rgb(arr):
        rgba = RdBu_r(v_norm(arr))
        return ['rgb({:.0f},{:.0f},{:.0f})'.format(c[0] * 255, c[1] * 255, c[2] * 255)
                for c in rgba]

    traj_colors = _to_rgb(traj_v)
    data_colors = _to_rgb(data_v)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=subplot_titles,
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
    all_x = np.concatenate([traj_x, data_x])
    all_y = np.concatenate([traj_y, data_y])
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
    r0_bf, _, _ = cartesian_to_spherical(geom_x, geom_y, geom_z)
    n_axis = get_axis_unit_vector(theta_axis_deg, phi_axis_deg)
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
        x=data_x, y=data_y, z=data_v, mode='markers',
        marker=dict(size=5, color=data_colors, symbol='circle'),
        error_x=dict(type='data', array=np.full_like(data_x, sigma_xy),
                     visible=True, color='gray', width=2),
        error_y=dict(type='data', array=np.full_like(data_y, sigma_xy),
                     visible=True, color='gray', width=2),
        error_z=dict(type='data', array=np.full_like(data_v, sigma_v),
                     visible=True, color='gray', width=2),
        showlegend=False,
    ), row=1, col=2)

    # Layout
    all_x = np.concatenate([traj_x, data_x])
    all_y = np.concatenate([traj_y, data_y])
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
            zaxis=dict(title='V_los', range=list(vlos_axis_range)),
            aspectmode='manual',
            aspectratio=dict(x=x_scale, y=y_scale, z=2),
        ),
        scene_camera=camera,
    )
    fig.write_html(output_html)
    return fig


def plot_orbit_combined(
    lobes,
    sigma_xy=60.0,
    sigma_v=1.331,
    v_range=(-6.0, 6.0),
    vlos_axis_range=(-20, 20),
    output_html='orbit_combined.html',
    subplot_titles=('PPP (best fit)', 'PPV (best fit)'),
):
    """Render PPP + PPV Plotly figure overlaying multiple lobes.

    Each lobe draws its own trajectory, observed data, rotation axis and
    equatorial plane in the shared PPP/PPV panels. All trajectories and data
    share a single V_los color normalization for direct color comparison.

    Parameters
    ----------
    lobes : list of dict
        One dict per lobe, with keys:
          traj_x, traj_y, traj_z, traj_v : arrays  (best-fit trajectory)
          data_x, data_y, data_v         : arrays  (observed PPV points)
          geom_x, geom_y, geom_z         : float   (starting position, sets r0)
          theta_axis_deg, phi_axis_deg   : float   (rotation-axis orientation)
          multi_trajs : list of (x, y, z, v) tuples, optional  (posterior samples)
    sigma_xy, sigma_v : float
        Observational uncertainties (error bars on PPV data points).
    v_range : (lo, hi)
        Color-scale range for V_los (RdBu_r), shared across all lobes.
    vlos_axis_range : (lo, hi)
        Fixed axis range for the V_los axis in the PPV panel.
    output_html : str
        Output HTML path.
    subplot_titles : (str, str)
        Titles for the PPP and PPV panels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])

    def _to_rgb(arr):
        rgba = RdBu_r(v_norm(np.asarray(arr)))
        return ['rgb({:.0f},{:.0f},{:.0f})'.format(c[0] * 255, c[1] * 255, c[2] * 255)
                for c in rgba]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    # Global spatial extent across all lobes (trajectories + data).
    all_x = np.concatenate(
        [np.asarray(l['traj_x']) for l in lobes]
        + [np.asarray(l['data_x']) for l in lobes]
    )
    all_y = np.concatenate(
        [np.asarray(l['traj_y']) for l in lobes]
        + [np.asarray(l['data_y']) for l in lobes]
    )

    # Central star (drawn once).
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'), showlegend=False,
    ), row=1, col=1)

    # Z=0 reference plane spanning all lobes.
    px_min, px_max = all_x.min(), all_x.max()
    py_min, py_max = all_y.min(), all_y.max()
    px = np.array([px_min, px_max, px_max, px_min])
    py = np.array([py_min, py_min, py_max, py_max])
    pz = np.zeros(4)
    fig.add_trace(go.Mesh3d(
        x=px, y=py, z=pz, i=[0, 0], j=[1, 3], k=[2, 2],
        color='gray', opacity=0.15, showlegend=False, name='Z=0',
    ), row=1, col=1)

    for lobe in lobes:
        traj_x = np.asarray(lobe['traj_x'])
        traj_y = np.asarray(lobe['traj_y'])
        traj_z = np.asarray(lobe['traj_z'])
        traj_v = np.asarray(lobe['traj_v'])
        data_x = np.asarray(lobe['data_x'])
        data_y = np.asarray(lobe['data_y'])
        data_v = np.asarray(lobe['data_v'])
        multi_trajs = lobe.get('multi_trajs') or []

        traj_colors = _to_rgb(traj_v)
        data_colors = _to_rgb(data_v)

        # PPP: posterior samples
        for tx, ty, tz, tv in multi_trajs:
            fig.add_trace(go.Scatter3d(
                x=tx, y=ty, z=tz, mode='lines',
                line=dict(color='gray', width=0.5), opacity=0.5, showlegend=False,
            ), row=1, col=1)

        # PPP: best-fit trajectory
        fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z, mode='lines+markers',
            marker=dict(size=3, color=traj_colors, opacity=0.8),
            line=dict(color='gray', width=2), showlegend=False,
        ), row=1, col=1)

        # Rotation axis + equatorial plane for this lobe.
        r0_bf, _, _ = cartesian_to_spherical(lobe['geom_x'], lobe['geom_y'], lobe['geom_z'])
        n_axis = get_axis_unit_vector(lobe['theta_axis_deg'], lobe['phi_axis_deg'])
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

        # PPV: best-fit trajectory
        fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_v, mode='lines+markers',
            marker=dict(size=3, color=traj_colors, opacity=0.8),
            line=dict(color='gray', width=2), showlegend=False,
        ), row=1, col=2)

        # PPV: observed data with error bars
        fig.add_trace(go.Scatter3d(
            x=data_x, y=data_y, z=data_v, mode='markers',
            marker=dict(size=5, color=data_colors, symbol='circle'),
            error_x=dict(type='data', array=np.full_like(data_x, sigma_xy, dtype=float),
                         visible=True, color='gray', width=2),
            error_y=dict(type='data', array=np.full_like(data_y, sigma_xy, dtype=float),
                         visible=True, color='gray', width=2),
            error_z=dict(type='data', array=np.full_like(data_v, sigma_v, dtype=float),
                         visible=True, color='gray', width=2),
            showlegend=False,
        ), row=1, col=2)

    # Layout scaling from global extent.
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
            zaxis=dict(title='V_los', range=list(vlos_axis_range)),
            aspectmode='manual',
            aspectratio=dict(x=x_scale, y=y_scale, z=2),
        ),
        scene_camera=camera,
    )
    fig.write_html(output_html)
    return fig


# ============================================================
# Standalone: set all model parameters, plot model vs. observation
# ============================================================

if __name__ == '__main__':
    # --- Model parameters (edit these) ---
    X = -440            # starting position [AU] 
    Y = -1000           # -440, -1000
    Z = -1928.2627             # -804
    V_R = -1.6682        # radial infall velocity [km/s]
    LOG10_OMEGA = -4.6757  # log10(rotation rate [round/yr])
    THETA_AXIS = 23.2817  # rotation-axis zenith [deg]
    PHI_AXIS = 121.9643    # rotation-axis azimuth [deg]
    M_STAR = 15.0         # central mass [M_sun]
    ALPHA = 1e7         # linear-drag coefficient
    T_SPAN = (0, 3000)    # integration time span [yr]
    N_EVAL = 1200
    STOPPING_R = 200.0    # stop when r crosses this [AU]
    AZIMUTH_MAX_DELTA_DEG = 230.0

    # --- Observation data ---
    OBS_NPZ = '../red-ppvf.npz'
    OUTPUT_HTML = 'orbit_view_nop_cluster3_south.html'
    OUTPUT_NPZ = 'orbit_view_nop_cluster3_south.npz'   # saved model trajectory (x, y, z, v_los)

    # --- Plot config ---
    SIGMA_XY = 60.0
    SIGMA_V = 1.331
    V_RANGE = (-6.0, 6.0)

    # --- Compute model trajectory ---
    traj_x, traj_y, traj_z, traj_v = compute_model_trajectory(
        X, Y, Z, V_R, LOG10_OMEGA, THETA_AXIS, PHI_AXIS,
        M=M_STAR, alpha=ALPHA,
        t_span=T_SPAN, n_eval=N_EVAL,
        stopping_r=STOPPING_R, azimuth_max_delta_deg=AZIMUTH_MAX_DELTA_DEG,
    )
    if traj_x is None:
        raise SystemExit('Trajectory computation failed (unbound orbit or solver error).')

    # --- Save model trajectory ---
    np.savez(OUTPUT_NPZ, x=traj_x, y=traj_y, z=traj_z, v_los=traj_v)
    print(f'Saved {OUTPUT_NPZ} ({traj_x.size} points)')

    # --- Observation ---
    _d = np.load(OBS_NPZ)
    data_x, data_y, data_v = _d['x'], _d['y'], _d['v']

    plot_orbit(
        traj_x, traj_y, traj_z, traj_v,
        data_x, data_y, data_v,
        X, Y, Z, THETA_AXIS, PHI_AXIS,
        sigma_xy=SIGMA_XY, sigma_v=SIGMA_V,
        v_range=V_RANGE,
        output_html=OUTPUT_HTML,
    )
    print(f'Saved {OUTPUT_HTML}')
