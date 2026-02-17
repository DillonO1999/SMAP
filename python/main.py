#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from lambert import lambert_universal, circular_velocity
from propagator import propagate_orbit

# --- 1. Constants & Mission Parameters ---
mu = 398600.4418
R_earth = 6378.137

# In main.py, try these values to verify the code works:
r1 = np.array([7000.0, 0.0, 0.0])
# Rotate r1 by 120 degrees to create a realistic r2
r2 = np.array([7000.0 * np.cos(np.radians(120)), 7000.0 * np.sin(np.radians(120)), 0.0])
tof = 3600 * 4.0  # 4 hours

# --- 2. Solve & Propagate ---
v1_trans, _ = lambert_universal(r1, r2, tof, mu)

# CHECK FOR FINITE VALUES BEFORE PROPAGATING
if np.any(np.isnan(v1_trans)):
    print("\n[!] MISSION ABORTED: Lambert solver failed to find a valid path.")
    print("Try increasing the Time of Flight (tof) or checking your r1/r2 coordinates.")
    # Exit gracefully or create an empty plot
    x_pts, y_pts, z_pts = [], [], [] 
else:
    t_points, state_history = propagate_orbit(r1, v1_trans, tof, mu)
    x_pts, y_pts, z_pts = state_history[0], state_history[1], state_history[2]

# --- 3. Interactive Plotly Visualization ---
fig = go.Figure()

# Create Earth Sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
ex = R_earth * np.outer(np.cos(u), np.sin(v))
ey = R_earth * np.outer(np.sin(u), np.sin(v))
ez = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))

# Add Earth to Plot
fig.add_trace(go.Surface(x=ex, y=ey, z=ez, colorscale='Blues', opacity=0.5, showscale=False, name="Earth"))

# Add Transfer Trajectory
fig.add_trace(go.Scatter3d(x=x_pts, y=y_pts, z=z_pts, mode='lines', 
                         line=dict(color='red', width=5), name="Transfer Orbit"))

# Add Departure and Arrival Points
fig.add_trace(go.Scatter3d(x=[r1[0]], y=[r1[1]], z=[r1[2]], mode='markers',
                         marker=dict(size=6, color='green'), name="Departure"))
fig.add_trace(go.Scatter3d(x=[r2[0]], y=[r2[1]], z=[r2[2]], mode='markers',
                         marker=dict(size=6, color='orange'), name="Arrival"))

# Scene Camera and Layout
fig.update_layout(
    title="SMAP Interactive Mission Planner",
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data' # Keeps the Earth spherical, not an egg
    ),
    margin=dict(r=0, l=0, b=0, t=40)
)

fig.show()