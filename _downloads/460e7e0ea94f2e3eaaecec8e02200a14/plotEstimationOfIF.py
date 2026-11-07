
"""
Estimation of the instantaneous frequency
=========================================

The code below estimate the instantanous frequency using the Discrete Wiener
Process Acceleration (DWPA) model.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objs as go

import ssm.inference
import ssm.kinematics.utils


#%%
# Set simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~

t0 = 0.0
tf = 200.0
fs = 50.0
alpha = 2 * np.pi * 0.01
beta = 2 * np.pi * 0.1
# beta = 0.0
omega_0 = 2 * np.pi * 0.01
# noise_snr = 100.0
noise_std = 0.1

#%%
# Set estimation parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~
pos_x0 = 0.0 # initial value for position
vel_x0 = 0.0 # initial value for velocity
ace_x0 = 0.0 # initial value for acceleration
sqrt_diag_V0_value = 1e-03 # square root of diagonal elements of the initial state covariance matrix
dt = 1e-3 # sampling interval
sigma_a = 1.0 # standard deviation of the state acceleration in the discrete Wiener process acceleration model https://github.com/joacorapela/ssm/blob/master/docs/kinematics/kinematics.pdf
sigma_x = 1.0 # standard deviation of the measurement position

#%%
# Simulate and plot a quadratic phase model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t = np.arange(t0, tf, 1.0/fs)
phi = beta * t**2 / 2 + omega_0 * t
phi_hat = phi + np.random.normal(loc=0.0,
                                 scale=noise_std,
                                 # scale=np.mean(phi)/noise_snr,
                                 size=len(phi))
true_IF = np.diff(phi) / dt

#%%
# Plot true and noise-corrupted phases
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

title = fr"$\varphi(t)=\frac{{\beta}}{2}t^2+\omega_0t+\epsilon,\beta={beta:.02f},\omega_0={omega_0:.02f}, \epsilon\sim\mathcal{{N}}(0,{noise_std}^2)$"
fig = go.Figure()
trace = go.Scatter(x=t, y=phi, name="True phase")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=phi_hat, name="Noise corrupted phase")
fig.add_trace(trace)
fig.update_xaxes(title="Time (sec)")
fig.update_yaxes(title="Phase")
fig.update_layout(title=title)

fig


#%%
# Set LDS parameters
# ~~~~~~~~~~~~~~~~~~

B, Q, Qe, Z, R = ssm.kinematics.utils.getLDSmatricesFor1DKinematics_np(
    dt=dt, sigma_a=sigma_a, pos_R_std=sigma_x)
m0 = np.array([pos_x0, vel_x0, ace_x0], dtype=np.double)
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Perform batch filtering
# ~~~~~~~~~~~~~~~~~~~~~~~
# View source code of `ssm.inference.filterLDS_SS_withMissingValues_np
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#filterLDS_SS_withMissingValues_np>`_

filterRes = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=phi_hat[np.newaxis, :], u=[0.0], B=B, Q=Q, m0=m0, V0=V0, a=[0.0], Z=Z, R=R)
    # y=phi_hat[np.newaxis, :], u=[0.0], B=B, Q=Q, m0=m0, V0=V0, a=[0.0], Z=Z, R=R)

#%%
# Perform batch smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~
# View source code of `ssm.inference.smoothLDS_SS
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#smoothLDS_SS>`_

smoothRes = ssm.inference.smoothLDS_SS(
    B=B, xnn=filterRes["xnn"], Pnn=filterRes["Pnn"],
    xnn1=filterRes["xnn1"], Pnn1=filterRes["Pnn1"], m0=m0, V0=V0)
mean_IF_LDS = smoothRes["xnN"][1,0,:]
std_IF_LDS = np.sqrt(smoothRes["PnN"][1,1,:])

#%%
# Estimate IF by finite differences
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
est_IF_fd = np.diff(phi_hat) / dt

#%%
# Plot true and estimated IF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

ci_upper = mean_IF_LDS + 1.96 * std_IF_LDS
ci_lower = mean_IF_LDS - 1.96 * std_IF_LDS

fig = go.Figure()
trace = go.Scatter(x=t, y=true_IF, name="True IF",
                   line=dict(color="rgb(255,0,0)"))
fig.add_trace(trace)
trace = go.Scatter(x=t, y=mean_IF_LDS, name="LDS",
                   line=dict(color="rgb(0,255,0)"),
                   legendgroup="LDS")
fig.add_trace(trace)
trace = go.Scatter(x=np.concatenate((t, t[::-1])),
                   y=np.concatenate((ci_upper, ci_lower[::-1])),
                   fill="toself",
                   fillcolor="rgba(0,255,0,0.3)",
                   line=dict(color='rgba(255,255,255,0)'),
                   hoverinfo="skip",
                   showlegend=False,
                   legendgroup="LDS")
fig.add_trace(trace)
trace = go.Scatter(x=t, y=est_IF_fd, name="Finite diff",
                   line=dict(color="rgb(0,0,255)"))
fig.add_trace(trace)
fig.update_xaxes(title="Time (sec)")
fig.update_yaxes(title="Instantaneous Frequency")
fig.update_layout(title=title)

fig

