
"""
Sampling from a Discrete Wiener Process Acceleration Model
==========================================================

The code below samples from a Discrete Wiener Process Acceleration (DWPA)
model.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objs as go

import ssm.tracking.utils
import ssm.simulation


#%%
# Set initial conditions and parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pos_x0 = 0.0
pos_y0 = 0.0
vel_x0 = 0.0
vel_y0 = 0.0
ace_x0 = 0.0
ace_y0 = 0.0
dt = 1e-3
num_pos = 1000
sigma_a = 1.0
sigma_x = 1.0
sigma_y = 1.0
sqrt_diag_V0_value = 1e-03

#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

B, Q, Qe, Z, R = ssm.tracking.utils.getLDSmatricesForKinematics_np(
    dt=dt, sigma_a=sigma_a, pos_x_R_std=sigma_x, pos_y_R_std=sigma_y)
R = np.diag([sigma_x**2, sigma_y**2])
m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~
# View source code of `ssm.simulation.simulateLDS
# <https://joacorapela.github.io/ssm/_modules/ssm/simulation.html#simulateLDS>`_

x0, x, y = ssm.simulation.simulateLDS(T=num_pos, B=B, Q=Q, Z=Z, R=R, m0=m0,
                                      V0=V0)

#%%
# Plot state positions and measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig = go.Figure()
trace_x = go.Scatter(x=x[0, :], y=x[3, :], mode="markers",
                     showlegend=True, name="state position")
trace_y = go.Scatter(x=y[0, :], y=y[1, :], mode="markers",
                     showlegend=True, name="measured position", opacity=0.3)
trace_start = go.Scatter(x=[x0[0]], y=[x0[3]], mode="markers",
                         text="initial state position", marker={"size": 7},
                         showlegend=False)
fig.add_trace(trace_x)
fig.add_trace(trace_y)
fig.add_trace(trace_start)
# fig_filename_pattern = "../../figures/simulated_pos.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig
