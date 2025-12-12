

"""
Learning and inference of latents with simple simulated data
============================================================

The code below learns and infers latents with simple simulated data.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objs as go

import ssm.simulation
import ssm.learning
import ssm.tracking.plotting

#%%
# Define variables for simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

m0 = np.array([0.0, 0.0], dtype=np.double)
V0 = np.array([[1e-3,0], [0,1e-3]], dtype=np.double)
B = np.array([[0.9872,-0.0272], [0.0080,1.0128]], dtype=np.double)
Q = np.array([[1e-3,0], [0,1e-3]], dtype=np.double)
Z = np.array([[1,0], [0,1]], dtype=np.double)
R = np.array([[.08,0], [0,.08]], dtype=np.double)
num_pos = 2000

#%%
# Perform simulation
# ~~~~~~~~~~~~~~~~~~

x0, x, y = ssm.simulation.simulateLDS(T=num_pos, B=B, Q=Q, Z=Z, R=R, m0=m0,
                                      V0=V0)
simulation_step = np.arange(x.shape[1])

#%%
# Plot simulation
# ~~~~~~~~~~~~~~~

fig = go.Figure()
trace_x = go.Scatter(x=x[0, :], y=x[1, :], mode="lines+markers",
                     showlegend=True, name="x")
trace_y = go.Scatter(x=y[0, :], y=y[1, :], mode="lines+markers",
                     showlegend=True, name="y", opacity=0.3)
trace_start = go.Scatter(x=[x0[0]], y=[x0[1]], mode="markers",
                         text="x0", marker={"size": 7},
                         showlegend=False)
fig.add_trace(trace_x)
fig.add_trace(trace_y)
fig.add_trace(trace_start)
fig

#%%
# Define initial conditions and control variables for EM learning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

m0_0 =y[:,0]
V0_0 = np.array([[1e-2,0], [0,1e-2]], np.double)
B0 = np.array([[1.0,-0.1], [0.0080,1.5]], np.double)
Q0 = np.array([[1e-4,0], [0,1e-2]], np.double)
Z0 = np.array([[1.0,0.1], [-0.1,1.0]], np.double)
R0 = np.array([[0.1,0], [0,0.1]], np.double)

# True Values
# V0_0 = np.array([[1e-3,0], [0,1e-3]], np.double)
# B0 = np.array([[.9872,-0.0272],[0.0080,1.0128]], np.double)
# Q0 = np.array([[1e-3,0], [0,1e-3]], np.double)
# Z0 = np.array([[1.0,0.0], [0.0,1.0]], np.double)
# R0 = np.array([[0.5,0], [0,0.5]], np.double)

max_iter = 50
tol = 1e-6
vars_to_estimate = {"B": True, "Q": True, "Z": True, "R": True,
                    "m0": True, "V0": True}

#%%
# Run EM
# ~~~~~~

optim_res = ssm.learning.em_SS_LDS(
    y=y, B0=B0, Q0=Q0, Z0=Z0, R0=R0, m0_0=m0_0, V0_0=V0_0,
    max_iter=max_iter, tol=tol, vars_to_estimate=vars_to_estimate,
)

#%%
# Plot log likelihood vs iteration number
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

N = len(optim_res["log_like"])
iter_no = np.arange(0, N)
fig = go.Figure()
trace = go.Scatter(x=iter_no,
                   y=optim_res["log_like"],
                   mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Iteration Number"),
                  yaxis=dict(title="Lower Bound"))
fig

#%%
# Filter simulations with the estimated parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

filter_res = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=optim_res["B"], Q=optim_res["Q"], m0=optim_res["m0"],
    V0=optim_res["V0"], Z=optim_res["Z"], R=optim_res["R"])

#%%
# Plot true and filtered states
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_values = x[(0, 1), :]
filtered_means = filter_res["xnn"][(0, 1), 0, :]
filtered_stds = np.sqrt(np.diagonal(a=filter_res["Pnn"], axis1=0, axis2=1)[:, (0, 1)].T)
fig = ssm.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
    time=simulation_step,
    ylabel="State",
    xlabel="Simulation Step",
    true_values=true_values,
    filtered_means=filtered_means,
    filtered_stds=filtered_stds)
fig.update_layout(title=f'Log-Likelihood: {filter_res["logLike"].squeeze()}')
fig

#%%
# Calculate the one-step ahead forecasts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

one_step_ahead_mean = optim_res["Z"] @ filter_res["xnn1"][:, 0, :]
# aux_covs = optim_res["R"] + (optim_res["Z"] @ filter_res["Pnn1"] @ optim_res["Z"].T)
aux1 = optim_res["Z"] @ filter_res["Pnn1"] # \in ijl
aux2 = optim_res["Z"].T # \in jk
aux3 = np.einsum("ijl,jk->ikl", aux1, aux2)
aux_covs = np.expand_dims(optim_res["R"], 2) + aux3
one_step_ahead_var = np.diagonal(aux_covs, axis1=0, axis2=1).T

#%%
# Plot measurements and one-step ahead forecasts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig = ssm.tracking.plotting.get_x_and_y_time_series_vs_time_fig(
    time=simulation_step,
    ylabel="One-Step Ahead Forecasts",
    xlabel="Simulation Step",
    measurements=y,
    filtered_means=one_step_ahead_mean,
    filtered_stds=np.sqrt(one_step_ahead_var))
fig

