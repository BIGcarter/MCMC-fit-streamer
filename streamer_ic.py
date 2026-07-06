import numpy as np
import astropy.units as u
from scipy.optimize import brentq


def get_axis_unit_vector(theta_axis_deg, phi_axis_deg):
    """
    Compute the unit vector of the rotation axis per custom convention.

    Parameters
    ----------
    theta_axis_deg : float
        Zenith from +y (0 deg) toward X-Z plane (0-180 deg).
    phi_axis_deg : float
        Azimuth in X-Z plane from +z (0 deg), counterclockwise (0-360 deg).

    Returns
    -------
    np.ndarray
        [nx, ny, nz] unit vector.
    """
    t = np.radians(theta_axis_deg)
    p = np.radians(phi_axis_deg)
    nx = -np.sin(t) * np.sin(p)
    ny = np.cos(t)
    nz = np.sin(t) * np.cos(p)
    return np.array([nx, ny, nz])


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian (x, y, z) to spherical coordinates under the model
    convention: theta from +y, phi from +z in the X-Z plane.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Returns
    -------
    r : float or ndarray
        Radial distance.
    theta_deg : float or ndarray
        Zenith angle from +y (degrees, 0~180°).
    phi_deg : float or ndarray
        Azimuth angle from +z in X-Z plane (degrees, 0~360°).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if np.isscalar(r) and r == 0:
        return 0.0, 0.0, 0.0
    mask = r > 0
    r_safe = np.where(mask, r, 1.0)
    theta = np.arccos(np.clip(y / r_safe, -1.0, 1.0))
    phi = np.arctan2(-x, z)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return r, np.degrees(theta), np.degrees(phi)


def build_local_frame(n_axis):
    """
    Build a 3x3 rotation matrix R whose columns are the local X', Y', Z' axes
    expressed in global coordinates. Z' aligns with n_axis.

    X_global = R @ X_local
    X_local = R.T @ X_global

    Parameters
    ----------
    n_axis : np.ndarray
        Unit vector defining the Z' axis.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    z_local = n_axis / np.linalg.norm(n_axis)

    # Choose a helper vector that is not nearly parallel to z_local
    if np.abs(z_local[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])

    x_local = np.cross(helper, z_local)
    x_local /= np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    R = np.column_stack((x_local, y_local, z_local))
    return R


def get_mendoza_initial_condition(
    r0, theta_part_deg, phi_part_deg, v_r, omega_round_yr,
    theta_axis_deg, phi_axis_deg
):
    """
    Mendoza model: radial infall with rotation around a custom axis.

    Parameters
    ----------
    r0 : float
        Initial radius (AU).
    theta_part_deg : float
        Zenith of the particle from +y (degrees, 0~180°).
    phi_part_deg : float
        Azimuth of the particle from +z in the X-Z plane (degrees, 0~360°).
    v_r : float
        Radial velocity (km/s, negative for infall).
    omega_round_yr : float
        Rotation speed (round/yr).
    theta_axis_deg : float
        Zenith of rotation axis from +y (degrees).
    phi_axis_deg : float
        Azimuth of rotation axis from +z (degrees).

    Returns
    -------
    tuple
        (x0, y0, z0, vx0, vy0, vz0) in AU and km/s.
    """
    # 1. Position in global Cartesian (same convention as get_axis_unit_vector)
    t_p = np.radians(theta_part_deg)
    p_p = np.radians(phi_part_deg)
    x0 = r0 * (-np.sin(t_p) * np.sin(p_p))
    y0 = r0 * np.cos(t_p)
    z0 = r0 * np.sin(t_p) * np.cos(p_p)
    r0_vec = np.array([x0, y0, z0])

    # 2. Unit conversions
    omega_rad_s = (omega_round_yr * u.cycle / u.yr).to(u.rad / u.s).value
    au_s_to_kms = (1.0 * u.au / u.s).to(u.km / u.s).value

    # 3. Rotation axis unit vector
    n_axis = get_axis_unit_vector(theta_axis_deg, phi_axis_deg)

    # 4. Radial velocity (toward origin)
    v_radial_vec = v_r * (r0_vec / r0)

    # 5. Rotation velocity: omega_vec x r0_vec
    v_rot_au_s = np.cross(omega_rad_s * n_axis, r0_vec)
    v_rot_vec = v_rot_au_s * au_s_to_kms

    v0_vec = v_radial_vec + v_rot_vec
    return (x0, y0, z0, v0_vec[0], v0_vec[1], v0_vec[2])


def get_ulrich_initial_condition(
    r0, theta_part_deg, phi_part_deg, GM, Rc,
    theta_axis_deg, phi_axis_deg
):
    """
    Ulrich model: ballistic infall from infinity, trajectory locked by Rc.

    Parameters
    ----------
    r0 : float
        Initial radius (AU).
    theta_part_deg : float
        Zenith of the particle from +y (degrees, 0~180°).
    phi_part_deg : float
        Azimuth of the particle from +z in the X-Z plane (degrees, 0~360°).
    GM : float
        Gravitational parameter in km^2/s^2 * AU.
    Rc : float
        Centrifugal radius (AU).
    theta_axis_deg : float
        Zenith of rotation axis from +y (degrees).
    phi_axis_deg : float
        Azimuth of rotation axis from +z (degrees).

    Returns
    -------
    tuple
        (x0, y0, z0, vx0, vy0, vz0) in AU and km/s.
    """
    # 1. Position in global Cartesian (same convention as get_axis_unit_vector)
    t_p = np.radians(theta_part_deg)
    p_p = np.radians(phi_part_deg)
    x0 = r0 * (-np.sin(t_p) * np.sin(p_p))
    y0 = r0 * np.cos(t_p)
    z0 = r0 * np.sin(t_p) * np.cos(p_p)
    r0_vec = np.array([x0, y0, z0])

    # 2. Project position into the local frame aligned with the rotation axis
    n_axis = get_axis_unit_vector(theta_axis_deg, phi_axis_deg)
    R = build_local_frame(n_axis)
    r0_local = R.T @ r0_vec

    # 3. Local spherical coordinates
    x_l, y_l, z_l = r0_local
    theta_local = np.arccos(z_l / r0)
    phi_local = np.arctan2(y_l, x_l)

    # Protect against sin(theta_local) near zero (poles)
    if np.sin(theta_local) < 1e-5:
        theta_local = 1e-5 if theta_local < np.pi / 2 else np.pi - 1e-5

    # Protect against the degenerate case where theta_local is exactly pi/2.
    # In this case the streamline equation r0/Rc = sin^2(theta_0) has no
    # real root when r0/Rc > 1, so we nudge theta_local by a tiny amount
    # to allow the root-finding to succeed with the (0, theta_local) bracket.
    if abs(theta_local - np.pi / 2) < 1e-6:
        theta_local = np.pi / 2 - 1e-6

    # 4. Solve for theta_0 (the polar angle at infinity) using brentq
    def streamline_root(theta_0):
        return (r0 / Rc) - (np.sin(theta_0)**2 * np.cos(theta_0)) / (
            np.cos(theta_0) - np.cos(theta_local)
        )

    if theta_local < np.pi / 2:
        theta_0_sol = brentq(streamline_root, 1e-7, theta_local - 1e-7)
    else:
        theta_0_sol = brentq(streamline_root, theta_local + 1e-7, np.pi - 1e-7)

    # 5. Ulrich velocity formulas in local spherical coordinates
    v_inf = np.sqrt(GM / r0)
    cos_ratio = np.cos(theta_local) / np.cos(theta_0_sol)
    v_r_l = -v_inf * np.sqrt(1.0 + cos_ratio)
    v_theta_l = (
        v_inf
        * ((np.cos(theta_0_sol) - np.cos(theta_local)) / np.sin(theta_local))
        * np.sqrt(1.0 + cos_ratio)
    )
    v_phi_l = (
        v_inf
        * (np.sin(theta_0_sol) / np.sin(theta_local))
        * np.sqrt(Rc / r0)
    )

    # 6. Convert local spherical velocity to local Cartesian velocity
    vx_l = (
        v_r_l * np.sin(theta_local) * np.cos(phi_local)
        + v_theta_l * np.cos(theta_local) * np.cos(phi_local)
        - v_phi_l * np.sin(phi_local)
    )
    vy_l = (
        v_r_l * np.sin(theta_local) * np.sin(phi_local)
        + v_theta_l * np.cos(theta_local) * np.sin(phi_local)
        + v_phi_l * np.cos(phi_local)
    )
    vz_l = v_r_l * np.cos(theta_local) - v_theta_l * np.sin(theta_local)
    v_local_vec = np.array([vx_l, vy_l, vz_l])

    # 7. Rotate velocity back to global Cartesian frame
    v0_vec = R @ v_local_vec

    return (x0, y0, z0, v0_vec[0], v0_vec[1], v0_vec[2])


def get_streamer_initial_state(model_type='mendoza', **kwargs):
    """
    Factory dispatch for streamer initial conditions.

    Parameters
    ----------
    model_type : str
        'mendoza' or 'ulrich'.
    **kwargs
        Model-specific parameters.

    Returns
    -------
    tuple
        (x0, y0, z0, vx0, vy0, vz0).
    """
    if model_type.lower() == 'mendoza':
        return get_mendoza_initial_condition(
            r0=kwargs['r0'],
            theta_part_deg=kwargs['theta_part_deg'],
            phi_part_deg=kwargs['phi_part_deg'],
            v_r=kwargs['v_r'],
            omega_round_yr=kwargs['omega_round_yr'],
            theta_axis_deg=kwargs['theta_axis_deg'],
            phi_axis_deg=kwargs['phi_axis_deg'],
        )
    elif model_type.lower() == 'ulrich':
        return get_ulrich_initial_condition(
            r0=kwargs['r0'],
            theta_part_deg=kwargs['theta_part_deg'],
            phi_part_deg=kwargs['phi_part_deg'],
            GM=kwargs['GM'],
            Rc=kwargs['Rc'],
            theta_axis_deg=kwargs['theta_axis_deg'],
            phi_axis_deg=kwargs['phi_axis_deg'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
