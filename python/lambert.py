#!/usr/bin/env python3

import numpy as np
from scipy.optimize import root_scalar

def stumpff_C(z):
    if abs(z) < 1e-8:
        return 1/2 - z/24 + z**2/720
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)

def stumpff_S(z):
    if abs(z) < 1e-8:
        return 1/6 - z/120 + z**2/5040
    elif z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
    else:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)

def lambert_universal(r1, r2, dt, mu):
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    cos_dtheta = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)
    
    if cos_dtheta < -0.99999999:
        r2 = r2 + np.array([0.1, 0, 0]) 
        r2_norm = np.linalg.norm(r2)
        cos_dtheta = np.dot(r1, r2) / (r1_norm * r2_norm)

    dtheta = np.arccos(cos_dtheta)
    A = np.sin(dtheta) * np.sqrt(r1_norm * r2_norm / (1 - cos_dtheta))
    
    if A == 0:
        return np.array([np.nan]*3), np.array([np.nan]*3)

    def time_of_flight_eq(z):
        C = stumpff_C(z)
        S = stumpff_S(z)

        if C <= 1e-5:
            return 1e15

        y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
        if y <= 0:
            # Return a progressive positive error penalizing unphysical states
            return 1e15 + abs(y) 

        chi = np.sqrt(y / C)
        dt_z = (chi**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return dt_z - dt

    # DYNAMIC BRACKETING: Locate where the sign change actually happens
    lower_bound = -40.0
    upper_bound = 39.4
    
    # Safely step the lower bound forward if it hits the unphysical y <= 0 region
    while time_of_flight_eq(lower_bound) > 1e14 and lower_bound < upper_bound:
        lower_bound += 0.5

    try:
        sol = root_scalar(time_of_flight_eq, bracket=[lower_bound, upper_bound], method='brentq')
        z = sol.root
    except:
        # Fallback to a broader physical search space if brentq gets stuck
        try:
            sol = root_scalar(time_of_flight_eq, bracket=[0.0, 39.4], method='brentq')
            z = sol.root
        except:
            return np.array([np.nan]*3), np.array([np.nan]*3)

    # Re-calculate final velocities
    C, S = stumpff_C(z), stumpff_S(z)
    y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
    
    f = 1 - y / r1_norm
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_norm

    return (r2 - f * r1) / g, (g_dot * r2 - r1) / g

def circular_velocity(r_vec, mu):
    r = np.linalg.norm(r_vec)
    v_mag = np.sqrt(mu / r)

    k_hat = np.array([0, 0, 1])
    t_hat = np.cross(k_hat, r_vec / r)

    return v_mag * t_hat

def transfer_delta_v(r1, r2, dt, mu):

    v1, v2 = lambert_universal(r1, r2, dt, mu)

    v1_circ = circular_velocity(r1, mu)
    v2_circ = circular_velocity(r2, mu)

    dv1 = np.linalg.norm(v1 - v1_circ)
    dv2 = np.linalg.norm(v2 - v2_circ)

    return dv1 + dv2