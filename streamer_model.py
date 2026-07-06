import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import solve_ivp


def gm_from_mstar(m_star):
    """Compute gravitational parameter GM in km²/s²·AU."""
    gm_physical = const.G * (m_star * u.M_sun)
    return gm_physical.to(u.km**2 / u.s**2 * u.au).value


def linear_drag(alpha=550.0):
    """
    Return a drag function for constant linear damping.

    Parameters
    ----------
    alpha : float
        Damping timescale. Larger alpha = weaker drag.

    Returns
    -------
    drag_func : callable
        Signature: drag_func(t, x, y, z, vx, vy, vz) -> (ax, ay, az)
    """
    def drag(t, x, y, z, vx, vy, vz):
        ax = -vx / alpha
        ay = -vy / alpha
        az = -vz / alpha
        return ax, ay, az
    return drag


def stopping_sphere(r_min):
    """
    Return an event function that terminates integration when the particle
    enters a sphere of radius r_min around the origin.

    Parameters
    ----------
    r_min : float
        Stopping radius (AU). Must be > 0.

    Returns
    -------
    event : callable
        Signature: event(t, Y, GM, drag_func) -> float.
        Has .terminal = True and .direction = -1 (trigger when crossing
        from above, i.e. r decreasing through r_min).
    """
    def event(t, Y, GM, drag_func):
        x, y, z = Y[0], Y[1], Y[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        return r - r_min

    event.terminal = True
    event.direction = -1
    return event


def eq_streamer(t, Y, GM, drag_func=None):
    """
    ODE right-hand side: gravity + optional drag.

    Parameters
    ----------
    t : float
        Time (independent variable).
    Y : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz].
    GM : float
        Gravitational parameter in km²/s²·AU.
    drag_func : callable or None
        drag_func(t, x, y, z, vx, vy, vz) -> (ax_drag, ay_drag, az_drag).
        If None, no drag is applied (pressureless).

    Returns
    -------
    dYdt : list of float, length 6
    """
    x, y, z, vx, vy, vz = Y
    r = np.sqrt(x**2 + y**2 + z**2)

    if r < 1e-3:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ax_grav = -GM * x / r**3
    ay_grav = -GM * y / r**3
    az_grav = -GM * z / r**3

    if drag_func is not None:
        ax_drag, ay_drag, az_drag = drag_func(t, x, y, z, vx, vy, vz)
    else:
        ax_drag = ay_drag = az_drag = 0.0

    return [vx, vy, vz,
            ax_grav + ax_drag,
            ay_grav + ay_drag,
            az_grav + az_drag]


def integrate_trajectory(initial_state, t_span, t_eval, GM, drag_func=None, events=None):
    """
    Integrate streamer ODE.

    Parameters
    ----------
    initial_state : sequence of 6 floats
        [x0, y0, z0, vx0, vy0, vz0].
    t_span : tuple (t_start, t_end)
    t_eval : np.ndarray
        Evaluation time points.
    GM : float
        Gravitational parameter in km²/s²·AU.
    drag_func : callable or None
        drag_func(t, x, y, z, vx, vy, vz) -> (ax_drag, ay_drag, az_drag).
    events : callable or list of callable or None
        Event function(s) for solve_ivp. Use stopping_sphere(r_min) to
        terminate when the particle reaches radius r_min.

    Returns
    -------
    solution : OdeResult
        solution.y has shape (6, n_points).
        If an event triggered, solution.t_events contains the event times.
    """
    return solve_ivp(eq_streamer, t_span, initial_state, args=(GM, drag_func),
                     t_eval=t_eval, method='RK45', events=events)
