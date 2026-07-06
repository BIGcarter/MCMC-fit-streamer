import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
from matplotlib.cm import RdBu_r

from streamer_ic import get_streamer_initial_state, cartesian_to_spherical, build_local_frame, get_axis_unit_vector
from streamer_model import gm_from_mstar, integrate_trajectory, linear_drag, stopping_sphere

# ============================================================
# Configuration
# ============================================================

M_star = 15.0                     # solar mass
GM = gm_from_mstar(M_star)

model_type = 'mendoza'             # 'mendoza' or 'ulrich'

cart_xyz = [-440, 1200, 500]
r0, theta0, phi0 = cartesian_to_spherical(cart_xyz[0],cart_xyz[1],cart_xyz[2])

# Common IC parameters
# ic_params = dict(
#     r0=500.0,
#     theta_part_deg=60.0,
#     phi_part_deg=45.0,
#     theta_axis_deg=0.0,
#     phi_axis_deg=0.0,
# )

ic_params = dict(
    r0=r0,
    theta_part_deg=theta0,
    phi_part_deg=phi0,
    theta_axis_deg=75.0,
    phi_axis_deg=110.0,
)

# Model-specific parameters
if model_type == 'mendoza':
    ic_params.update(dict(
        v_r=-4.0,
        omega_round_yr=4e-5,  # 8e-5
    ))
elif model_type == 'ulrich':
    ic_params.update(dict(
        GM=GM,
        Rc=100.0,
    ))

# Integration settings
drag_func = linear_drag(alpha=500)
stopping_r = 150.0   # AU — terminate when particle reaches this radius
t_span = (0, 1500)
t_eval = np.linspace(t_span[0], t_span[1], 1200)

# ============================================================
# Compute
# ============================================================

x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state(model_type, **ic_params)
print(f'Model: {model_type}')
print(f'Initial position: ({x0:.1f}, {y0:.1f}, {z0:.1f}) AU')
print(f'Initial velocity: ({vx0:.2f}, {vy0:.2f}, {vz0:.2f}) km/s')

initial_state = [x0, y0, z0, vx0, vy0, vz0]
sol = integrate_trajectory(initial_state, t_span, t_eval, GM, drag_func,
                           events=stopping_sphere(stopping_r))
print(f'Integration: {sol.message}')
if sol.t_events and len(sol.t_events[0]) > 0:
    t_stop = sol.t_events[0][0]
    r_stop = np.sqrt(sol.y_events[0][0][0]**2
                     + sol.y_events[0][0][1]**2
                     + sol.y_events[0][0][2]**2)
    print(f'Stopped at t={t_stop:.1f}, r={r_stop:.1f} AU (hit r_min={stopping_r})')
else:
    print(f'Completed full integration without hitting r_min={stopping_r}')

x_orb, y_orb, z_orb = sol.y[0], sol.y[1], sol.y[2]
V_los = sol.y[5]  # vz as line-of-sight velocity

# Shared color normalization for V_los
v_range = [-20, 20]
v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])
_v_colors_rgba = RdBu_r(v_norm(V_los))
v_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(
    c[0]*255, c[1]*255, c[2]*255) for c in _v_colors_rgba]


# ============================================================
# load observation data
# ============================================================
# peak = np.load("../ppvf.npz")
peak_blue = np.load("../blue-ppvf.npz")
peak_red = np.load("../red-ppvf.npz")

# ============================================================
# Visualization
# ============================================================

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('PPP', 'PPV'),
    horizontal_spacing=0.08,
)

# --- Left: XYZ position space ---

fig.add_trace(go.Scatter3d(
    x=x_orb, y=y_orb, z=z_orb,
    mode='lines+markers',
    marker=dict(
        size=3,
        color=v_colors,
        opacity=0.8,
        showscale=False,
    ),
    line=dict(color='gray', width=2),
    showlegend=False,
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=8, color='black', symbol='diamond'),
    showlegend=False,
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode='markers',
    marker=dict(size=8, color='yellow', symbol='square'),
    showlegend=False,
), row=1, col=1)

fig.add_trace(go.Cone(
    x=[x0], y=[y0], z=[z0],
    u=[vx0], v=[vy0], w=[vz0],
    sizemode='absolute',
    sizeref=1.5,
    colorscale=[[0, 'gold'], [1, 'gold']],
    showscale=False,
    anchor='tail',
), row=1, col=1)

# Rotation axis and equatorial plane
n_axis = get_axis_unit_vector(ic_params['theta_axis_deg'],
                                ic_params['phi_axis_deg'])
R_plane = build_local_frame(n_axis)
plane_r = 0.5 * r0  # disc radius
n_ring = 80
theta_ring = np.linspace(0, 2 * np.pi, n_ring)
ring_local = np.column_stack([
    plane_r * np.cos(theta_ring),
    plane_r * np.sin(theta_ring),
    np.zeros(n_ring),
])
ring_global = (R_plane @ ring_local.T).T  # (n_ring, 3)

tri_i, tri_j, tri_k = [], [], []
for i in range(n_ring - 1):
    tri_i.append(0)
    tri_j.append(i + 1)
    tri_k.append(i + 2)
tri_i.append(0); tri_j.append(n_ring); tri_k.append(1)

disc_vertices = np.vstack([[0, 0, 0], ring_global])

fig.add_trace(go.Mesh3d(
    x=disc_vertices[:, 0], y=disc_vertices[:, 1], z=disc_vertices[:, 2],
    i=tri_i, j=tri_j, k=tri_k,
    color='lightblue', opacity=0.25,
    showlegend=False,
), row=1, col=1)

# Rotation axis arrow (line + cone head)
axis_len = 0.3 * r0
axis_tip = n_axis * axis_len
fig.add_trace(go.Scatter3d(
    x=[0, axis_tip[0]], y=[0, axis_tip[1]], z=[0, axis_tip[2]],
    mode='lines',
    line=dict(color='green', width=4),
    showlegend=False,
), row=1, col=1)

fig.add_trace(go.Cone(
    x=[axis_tip[0]], y=[axis_tip[1]], z=[axis_tip[2]],
    u=[n_axis[0]], v=[n_axis[1]], w=[n_axis[2]],
    sizemode='absolute',
    sizeref=axis_len * 0.15,
    colorscale=[[0, 'green'], [1, 'green']],
    showscale=False,
    anchor='tip',
), row=1, col=1)


# x0, y0, z0, vx0, vy0, vz0


# --- Right: XY–V_los (PPV space) ---

fig.add_trace(go.Scatter3d(
    x=x_orb, y=y_orb, z=V_los,
    mode='lines+markers',
    marker=dict(
        size=3,
        color=v_colors,
        opacity=0.8,
        showscale=False,
    ),
    line=dict(color='gray', width=2),
    showlegend=False,
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=8, color='black', symbol='diamond'),
    showlegend=False,
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[vz0],
    mode='markers',
    marker=dict(size=6, color='yellow', symbol='square'),
    showlegend=False,
), row=1, col=2)

_blue_rgba = RdBu_r(v_norm(peak_blue['v']))
blue_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(
    c[0]*255, c[1]*255, c[2]*255) for c in _blue_rgba]

_red_rgba = RdBu_r(v_norm(peak_red['v']))
red_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(
    c[0]*255, c[1]*255, c[2]*255) for c in _red_rgba]

fig.add_trace(go.Scatter3d(
    x=peak_blue['x'], y=peak_blue['y'], z=peak_blue['v'],
    mode='markers',
    marker=dict(size=5, color=blue_colors, symbol='circle'),
    showlegend=False,
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=peak_red['x'], y=peak_red['y'], z=peak_red['v'],
    mode='markers',
    marker=dict(size=5, color=red_colors, symbol='diamond'),
    showlegend=False,
), row=1, col=2)

# --- Layout ---

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=-2.5),
)

all_x = np.concatenate([peak_blue['x'], peak_red['x']])
all_y = np.concatenate([peak_blue['y'], peak_red['y']])
x_range = all_x.max() - all_x.min()
y_range = all_y.max() - all_y.min()
max_spatial_range = max(x_range, y_range)
x_scale = x_range / max_spatial_range
y_scale = y_range / max_spatial_range
z_scale = 2

fig.update_layout(
    template='plotly_white',
    showlegend=False,
    scene1=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data',
    ),
    scene2=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='V_los', range=[-20, 20]),
        aspectmode='manual',
        aspectratio=dict(x=x_scale, y=y_scale, z=z_scale),
    ),
    scene_camera=camera,
)


# ======================================
# plot obs data
# ======================================



fig.show()
