import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from astropy.io import fits
from plotly.subplots import make_subplots
import astropy.units as u
import astropy.constants as const

from astropy import wcs



M_star = 15.0  # solar mass

GM_physical = const.G * (M_star * u.M_sun)
GM = GM_physical.to(u.km**2 / u.s**2 * u.au).value

##############
# Pressureless
##############

# some eqs
# def eq_streamer(t, Y, GM):
#     x, y, z, vx, vy, vz = Y
#     r = np.sqrt(x**2 + y**2 + z**2)
#     if r < 1e-3:
#         return [0, 0, 0, 0, 0, 0]
#     ax, ay, az = -GM * x / r**3, -GM * y / r**3, -GM * z / r**3
#     return [vx, vy, vz, ax, ay, az]

##############
# Linear damping
##############

def eq_streamer(t, Y, GM, alpha=550.0):
    x, y, z, vx, vy, vz = Y
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-3:
        return [0, 0, 0, 0, 0, 0]
    
    ax_grav = -GM * x / r**3
    ay_grav = -GM * y / r**3
    az_grav = -GM * z / r**3
    
    ax_drag = -vx / alpha
    ay_drag = -vy / alpha
    az_drag = -vz / alpha
    
    ax = ax_grav + ax_drag
    ay = ay_grav + ay_drag
    az = az_grav + az_drag
    
    return [vx, vy, vz, ax, ay, az]


x0, y0, z0 = 440, 1200, 500.0   # au
vx0, vy0, vz0 = -4, -2, -2   # km/s

initial_state = [x0, y0, z0, vx0, vy0, vz0]
t_span = (0, 500)                    
t_eval = np.linspace(t_span[0], t_span[1], 1200)

solution = solve_ivp(eq_streamer, t_span, initial_state, args=(GM,), t_eval=t_eval, method='RK45')

x_orb, y_orb, z_orb = solution.y[0], solution.y[1], solution.y[2]
vx_orb, vy_orb, vz_orb = solution.y[3], solution.y[4], solution.y[5]

RA = x_orb
Dec = y_orb
V_los = vz_orb 


# fig = make_subplots(
#     rows=2, cols=1,
#     specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
#     vertical_spacing=0.05
# )

fig = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'scene'}]],
    # vertical_spacing=0.05
)

#PPP
fig.add_trace(go.Scatter3d(
    x=x_orb, y=y_orb, z=z_orb,
    mode='lines+markers',
    marker=dict(
        size=3,
        color=V_los,          
        colorscale='RdBu_r',
        opacity=0.8,
        showscale=False       
    ),
    line=dict(color='gray', width=2),
    showlegend=False
), row=1, col=1)

# central star
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=8, color='black', symbol='diamond'),
    showlegend=False
), row=1, col=1)

# Starting point
fig.add_trace(go.Scatter3d(
    x=[x0], y=[y0], z=[z0],
    mode='markers',
    marker=dict(size=8, color='yellow', symbol='square'),
    showlegend=False
), row=1, col=1)

# Inital velocity
fig.add_trace(go.Cone(
    x=[x0], y=[y0], z=[z0],          
    u=[vx0], v=[vy0], w=[vz0],       
    sizemode="absolute",             
    sizeref=1.5,                     
    colorscale=[[0, 'gold'], [1, 'gold']], 
    showscale=False,
    anchor="tail"                    
), row=1, col=1)

# PPV
# fig.add_trace(go.Scatter3d(
#     x=RA, y=Dec, z=V_los,   
#     mode='lines+markers',
#     marker=dict(
#         size=3,
#         color=V_los,          
#         colorscale='RdBu_r',  
#         opacity=0.8,
#         colorbar=dict(thickness=15, len=0.4, y=0.25)  
#     ),
#     line=dict(color='gray', width=2),
#     showlegend=False
# ), row=2, col=1)

# # central star
# fig.add_trace(go.Scatter3d(
#     x=[0], y=[0], z=[0],
#     mode='markers',
#     marker=dict(size=8, color='black', symbol='diamond'),
#     showlegend=False
# ), row=2, col=1)

# # starting point
# fig.add_trace(go.Scatter3d(
#     x=[x0], y=[y0], z=[vz0],
#     mode='markers',
#     marker=dict(size=8, color='yellow', symbol='square'),
#     showlegend=False
# ), row=2, col=1)


camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=-2.5)
)

fig.update_layout(
    template="plotly_white",
    # margin=dict(l=0, r=0, b=0, t=10),
    showlegend=False,
    
    scene1=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data'
    ),

    scene_camera = camera
    
    # scene2=dict(
    #     xaxis=dict(title='X'),
    #     yaxis=dict(title='Y'),
    #     zaxis=dict(title='V'),
    #     aspectmode='data'
    # )
)

fig.show()