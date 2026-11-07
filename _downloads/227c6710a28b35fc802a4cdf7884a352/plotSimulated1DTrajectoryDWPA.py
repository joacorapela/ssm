
"""
Sampling from a 1D Discrete Wiener Process Acceleration Model
=============================================================

The code below samples from a 1D Discrete Wiener Process Acceleration (DWPA)
model.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objs as go

import ssm.kinematics.utils
import ssm.simulation


#%%
# Set initial conditions and parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pos_x0 = 0.0
vel_x0 = 0.0
ace_x0 = 0.0
dt = 1e-3
num_pos = 1000
sigma_a = 1.0
sigma_x = 1.0
sqrt_diag_V0_value = 1e-03

#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

B, Q, Qe, Z, R = ssm.kinematics.utils.getLDSmatricesFor1DKinematics_np(
    dt=dt, sigma_a=sigma_a, pos_R_std=sigma_x)
R = np.diag([sigma_x**2,])
m0 = np.array([pos_x0, vel_x0, ace_x0],
              dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Sample from the LDS
# ~~~~~~~~~~~~~~~~~~~
# View source code of `ssm.simulation.simulateLDS
# <https://joacorapela.github.io/ssm/_modules/ssm/simulation.html#simulateLDS>`_

M = B.shape[0]
N = Z.shape[0]
u = np.zeros(shape=(M, 1), dtype=np.double)
a = np.zeros(shape=(N, 1), dtype=np.double)
x0, x, y = ssm.simulation.simulateLDS(T=num_pos, u=u, B=B, Q=Q, m0=m0, V0=V0,
                                      a=a, Z=Z, R=R)
t = np.arange(num_pos) * dt

#%%
# Plot state positions and measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig = go.Figure()
trace = go.Scatter(x=t, y=x[0, :], mode="markers",
                   showlegend=True, name="state position")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=x[1, :], mode="markers",
                   showlegend=True, name="state velocity")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=x[2, :], mode="markers",
                   showlegend=True, name="state acceleration")
fig.add_trace(trace)
trace = go.Scatter(x=[t[0]], y=[x0[0]], mode="markers",
                   text="initial state position", marker={"size": 7},
                   showlegend=False)
fig.add_trace(trace)
trace = go.Scatter(x=t, y=y[0, :], mode="markers",
                   showlegend=True, name="measured position")
fig.add_trace(trace)
fig.update_xaxes(title="Time (sec)")
fig

# fig_filename_pattern = "../../../figures/simulated_pos.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))

# breakpoint()
