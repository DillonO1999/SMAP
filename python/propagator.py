#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp

def orbit_dynamics(t, y, mu, J2=0, R_earth=6378.137):
    """
    State-space equations for orbital motion.
    y = [x, y, z, vx, vy, vz]
    """
    r_vec = y[:3]
    v_vec = y[3:]
    r = np.linalg.norm(r_vec)
    
    # Fundamental Two-Body Acceleration
    a_mag = -mu / r**3
    a_2body = a_mag * r_vec
    
    # J2 Perturbation Acceleration
    a_j2 = np.zeros(3)
    if J2 != 0:
        z2 = r_vec[2]**2
        r2 = r**2
        factor = (3/2) * J2 * (mu / r2) * (R_earth / r)**2
        
        a_j2[0] = r_vec[0] / r * (5 * z2 / r2 - 1)
        a_j2[1] = r_vec[1] / r * (5 * z2 / r2 - 1)
        a_j2[2] = r_vec[2] / r * (5 * z2 / r2 - 3)
        a_j2 *= factor

    return np.concatenate([v_vec, a_2body + a_j2])

def propagate_orbit(r0, v0, duration, mu, step_size=60, J2=0.0010826):
    """
    Integrates the orbit over a given duration (seconds).
    """
    y0 = np.concatenate([r0, v0])
    t_span = (0, duration)
    t_eval = np.arange(0, duration, step_size)
    
    sol = solve_ivp(
        orbit_dynamics, 
        t_span, 
        y0, 
        args=(mu, J2), 
        t_eval=t_eval, 
        rtol=1e-9, 
        atol=1e-12
    )
    
    return sol.t, sol.y