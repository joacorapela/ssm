
"""
Comparison between EM and gradient ascent for tracking a foraging mouse
=======================================================================

The code below compares the expectation maximization (EM) and gradient ascent
algorithm for tracking a foraging mouse.


"""

import sys
import os.path
import argparse
import configparser
import math
import random
import pickle
import numpy as np
import pandas as pd
import torch
import scipy
import plotly.graph_objects as go

import ssm.learning


#%%
# Define parameters for estimation
# --------------------------------

skip_estimation_sigma_a = False
skip_estimation_R = False
skip_estimation_m0 = False
skip_estimation_V0 = False

start_position = 0
# number_positions = 10000
# number_positions = 7500
number_positions = 50
lbfgs_max_iter = 2
lbfgs_tolerance_grad = -1
lbfgs_tolerance_change = 1e-3
lbfgs_lr = 1.0
lbfgs_n_epochs = 100
lbfgs_tol = 1e-3
em_max_iter = 200
Qe_reg_param_ga = None
Qe_reg_param_em = 1e-5

#%%
# Provide initial conditions
# --------------------------

pos_x0 = 0.0
pos_y0 = 0.0
vel_x0 = 0.0
vel_y0 = 0.0
ace_x0 = 0.0
ace_y0 = 0.0
sigma_a0 = 1.0
sigma_x0 = 1.0
sigma_y0 = 1.0
sqrt_diag_V0_value = 0.1

if math.isnan(pos_x0):
    pos_x0 = y[0, 0]
if math.isnan(pos_y0):
    pos_y0 = y[1, 0]

#%%
# Get mouse positions
# -------------------

data_filename = "http://www.gatsby.ucl.ac.uk/~rapela/svGPFA/data/positions_session003_start0.00_end15548.27.csv"
data = pd.read_csv(data_filename)
data = data.iloc[start_position:start_position+number_positions,:]
y = np.transpose(data[["x", "y"]].to_numpy())
date_times = pd.to_datetime(data["time"])
dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()

#%%
# Build the matrices of the CWPA model
# ------------------------------------

B, _, Qe, Z, _ = ssm.tracking.utils.getLDSmatricesForKinematics_np(
    dt=dt, sigma_a=np.nan, pos_x_R_std=np.nan, pos_y_R_std=np.nan)
m0 = np.array([pos_x0, vel_x0, ace_x0, pos_y0, vel_y0, ace_y0],
              dtype=np.double)

vars_to_estimate = {}
if skip_estimation_sigma_a:
    vars_to_estimate["sigma_a"] = False
else:
    vars_to_estimate["sigma_a"] = True

if skip_estimation_R:
    vars_to_estimate["pos_x_R_std"] = False
    vars_to_estimate["pos_y_R_std"] = False
    vars_to_estimate["R"] = False
else:
    vars_to_estimate["pos_x_R_std"] = True
    vars_to_estimate["pos_y_R_std"] = True
    vars_to_estimate["R"] = True

if skip_estimation_m0:
    vars_to_estimate["m0"] = False
else:
    vars_to_estimate["m0"] = True

if skip_estimation_V0:
    vars_to_estimate["sqrt_diag_V0"] = False
    vars_to_estimate["V0"] = False
else:
    vars_to_estimate["sqrt_diag_V0"] = True
    vars_to_estimate["V0"] = True

#%%
# Perform gradient ascent optimization
# ------------------------------------

sqrt_diag_R_torch = torch.DoubleTensor([sigma_x0, sigma_y0])
m0_torch = torch.from_numpy(m0.copy())
sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                         for i in range(len(m0))])
if Qe_reg_param_ga is not None:
    Qe_regularized_ga = Qe + Qe_reg_param_ga * np.eye(Qe.shape[0])
else:
    Qe_regularized_ga = Qe
y_torch = torch.from_numpy(y.astype(np.double))
B_torch = torch.from_numpy(B.astype(np.double))
Qe_regularized_ga_torch = torch.from_numpy(Qe_regularized_ga.astype(np.double))
Z_torch = torch.from_numpy(Z.astype(np.double))

optim_res_ga = ssm.learning.torch_lbfgs_optimize_SS_tracking_diagV0(
    y=y_torch, B=B_torch, Qe=Qe_regularized_ga_torch, Z=Z_torch,
    sigma_a0=sigma_a0, pos_x_R_std0=sigma_x0, pos_y_R_std0=sigma_y0, m0_0=m0_torch,
    sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=lbfgs_max_iter, lr=lbfgs_lr,
    vars_to_estimate=vars_to_estimate, tolerance_grad=lbfgs_tolerance_grad,
    tolerance_change=lbfgs_tolerance_change, n_epochs=lbfgs_n_epochs,
    tol=lbfgs_tol)

print("gradient ascent: " + optim_res_ga["termination_info"])

#%%
# Perform EM optimization
# -----------------------

sqrt_diag_R = np.array([sigma_x0, sigma_y0])
R_0 = np.diag(sqrt_diag_R)
m0_0 = m0
sqrt_diag_V0 = np.array([sqrt_diag_V0_value for i in range(len(m0))])
V0_0 = np.diag(sqrt_diag_V0)

times = np.arange(0, y.shape[1]*dt, dt)
not_nan_indices_y0 = set(np.where(np.logical_not(np.isnan(y[0, :])))[0])
not_nan_indices_y1 = set(np.where(np.logical_not(np.isnan(y[1, :])))[0])
not_nan_indices = np.array(sorted(not_nan_indices_y0.union(not_nan_indices_y1)))
y_no_nan = y[:, not_nan_indices]
t_no_nan = times[not_nan_indices]
y_interpolated = np.empty_like(y)
tck, u = scipy.interpolate.splprep([y_no_nan[0, :], y_no_nan[1, :]], s=0, u=t_no_nan)
y_interpolated[0, :], y_interpolated[1, :] = scipy.interpolate.splev(times, tck)

if Qe_reg_param_em is not None:
    Qe_regularized_em = Qe + Qe_reg_param_em * np.eye(Qe.shape[0])
else:
    Qe_regularized_em = Qe

optim_res_em  = ssm.learning.em_SS_tracking(
    y=y_interpolated, B=B, sigma_a0=sigma_a0,
    Qe=Qe_regularized_em, Z=Z, R_0=R_0, m0_0=m0_0, V0_0=V0_0,
    vars_to_estimate=vars_to_estimate,
    max_iter=em_max_iter)

print("EM: " + optim_res_em["termination_info"])

#%%
# Plot convergence
# ----------------

fig = go.Figure()
trace = go.Scatter(x=optim_res_ga["elapsed_time"], y=optim_res_ga["log_like"],
                  name="Gradient ascent", mode="lines+markers")
fig.add_trace(trace)
trace = go.Scatter(x=optim_res_em["elapsed_time"], y=optim_res_em["log_like"],
                   name="EM", mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis_title="Elapsed Time (sec)",
                  yaxis_title="Log Likelihood")
fig

#%%
# Perform smoothing with optimized parameters
# -------------------------------------------

#%%
# Gradient ascent
# ~~~~~~~~~~~~~~~

#%%
# Perform batch filtering
# #######################
# View source code of `ssm.inference.filterLDS_SS_withMissingValues_np
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#filterLDS_SS_withMissingValues_np>`_

Q_ga = optim_res_ga["estimates"]["sigma_a"].item()**2*Qe
m0_ga = optim_res_ga["estimates"]["m0"].numpy()
V0_ga = np.diag(optim_res_ga["estimates"]["sqrt_diag_V0"].numpy()**2)
R_ga = np.diag([optim_res_ga["estimates"]["pos_x_R_std"].item()**2,
                optim_res_ga["estimates"]["pos_y_R_std"].item()**2])

filterRes_ga = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=B, Q=Q_ga, m0=m0_ga, V0=V0_ga, Z=Z, R=R_ga)

#%%
# Perform batch smoothing
# #######################
# View source code of `ssm.inference.smoothLDS_SS
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#smoothLDS_SS>`_

smoothRes_ga = ssm.inference.smoothLDS_SS(
    B=B, xnn=filterRes_ga["xnn"], Pnn=filterRes_ga["Pnn"],
    xnn1=filterRes_ga["xnn1"], Pnn1=filterRes_ga["Pnn1"], m0=m0_ga, V0=V0_ga)

#%%
# EM
# ~~

#%%
# Perform batch filtering
# #######################
# View source code of `ssm.inference.filterLDS_SS_withMissingValues_np
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#filterLDS_SS_withMissingValues_np>`_

Q_em = optim_res_em["estimates"]["sigma_a"].item()**2*Qe
m0_em = optim_res_em["estimates"]["m0"]
V0_em = optim_res_em["estimates"]["V0"]
R_em = optim_res_em["estimates"]["R"]

filterRes_em = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=y, B=B, Q=Q_em, m0=m0_em, V0=V0_em, Z=Z, R=R_em)

#%%
# Perform batch smoothing
# #######################
# View source code of `ssm.inference.smoothLDS_SS
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#smoothLDS_SS>`_

smoothRes_em = ssm.inference.smoothLDS_SS(
    B=B, xnn=filterRes_em["xnn"], Pnn=filterRes_em["Pnn"],
    xnn1=filterRes_em["xnn1"], Pnn1=filterRes_em["Pnn1"], m0=m0_em, V0=V0_em)

#%%
# Plot smoothing results
# ----------------------

#%%
# Define function for plotting
# ############################

def get_fig_kinematics_vs_time(
    time,
    measured_x, measured_y,
    finite_diff_x, finite_diff_y,
    ga_mean_x, ga_mean_y,
    ga_ci_x_upper, ga_ci_y_upper,
    ga_ci_x_lower, ga_ci_y_lower,
    em_mean_x, em_mean_y,
    em_ci_x_upper, em_ci_y_upper,
    em_ci_x_lower, em_ci_y_lower,
    cb_alpha,
    color_true,
    color_measured,
    color_finite_diff,
    color_ga_pattern,
    color_em_pattern,
    xlabel, ylabel):

    fig = go.Figure()
    if measured_x is not None:
        trace_mes_x = go.Scatter(
            x=time, y=measured_x,
            mode="markers",
            marker={"color": color_measured},
            name="measured x",
            showlegend=True,
        )
        fig.add_trace(trace_mes_x)
    if measured_y is not None:
        trace_mes_y = go.Scatter(
            x=time, y=measured_y,
            mode="markers",
            marker={"color": color_measured},
            name="measured y",
            showlegend=True,
        )
        fig.add_trace(trace_mes_y)
    if finite_diff_x is not None:
        trace_fd_x = go.Scatter(
            x=time, y=finite_diff_x,
            mode="markers",
            marker={"color": color_finite_diff},
            name="finite difference x",
            showlegend=True,
        )
        fig.add_trace(trace_fd_x)
    if finite_diff_y is not None:
        trace_fd_y = go.Scatter(
            x=time, y=finite_diff_y,
            mode="markers",
            marker={"color": color_finite_diff},
            name="finite difference y",
            showlegend=True,
        )
        fig.add_trace(trace_fd_y)
    trace_ga_x = go.Scatter(
        x=time, y=ga_mean_x,
        mode="markers",
        marker={"color": color_ga_pattern.format(1.0)},
        name="grad. ascent x",
        showlegend=True,
        legendgroup="ga_x",
    )
    fig.add_trace(trace_ga_x)
    trace_ga_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([ga_ci_x_upper, ga_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_ga_pattern.format(cb_alpha),
        line=dict(color=color_ga_pattern.format(0.0)),
        showlegend=False,
        legendgroup="ga_x",
    )
    fig.add_trace(trace_ga_x_cb)
    trace_ga_y = go.Scatter(
        x=time, y=ga_mean_y,
        mode="markers",
        marker={"color": color_ga_pattern.format(1.0)},
        name="grad. ascent y",
        showlegend=True,
        legendgroup="ga_y",
    )
    fig.add_trace(trace_ga_y)
    trace_ga_y_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([ga_ci_y_upper, ga_ci_y_lower[::-1]]),
        fill="toself",
        fillcolor=color_ga_pattern.format(cb_alpha),
        line=dict(color=color_ga_pattern.format(0.0)),
        showlegend=False,
        legendgroup="ga_y",
    )
    fig.add_trace(trace_ga_y_cb)
    trace_em_x = go.Scatter(
        x=time, y=em_mean_x,
        mode="markers",
        marker={"color": color_em_pattern.format(1.0)},
        name="EM x",
        showlegend=True,
        legendgroup="em_x",
    )
    fig.add_trace(trace_em_x)
    trace_em_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([em_ci_x_upper, em_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_em_pattern.format(cb_alpha),
        line=dict(color=color_em_pattern.format(0.0)),
        showlegend=False,
        legendgroup="em_x",
    )
    fig.add_trace(trace_em_x_cb)
    trace_em_y = go.Scatter(
        x=time, y=em_mean_y,
        mode="markers",
        marker={"color": color_em_pattern.format(1.0)},
        name="EM y",
        showlegend=True,
        legendgroup="em_y",
    )
    fig.add_trace(trace_em_y)
    trace_em_y_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([em_ci_y_upper, em_ci_y_lower[::-1]]),
        fill="toself",
        fillcolor=color_em_pattern.format(cb_alpha),
        line=dict(color=color_em_pattern.format(0.0)),
        showlegend=False,
        legendgroup="em_y",
    )
    fig.add_trace(trace_em_y_cb)

    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                     )
    return fig

#%%
# Set variables for plotting
# ##########################

N = y.shape[1]
time = np.arange(0, N*dt, dt)
smoothed_means_ga = smoothRes_ga["xnN"]
smoothed_covs_ga = smoothRes_ga["PnN"]
smoothed_std_x_y_ga = np.sqrt(np.diagonal(a=smoothed_covs_ga, axis1=0, axis2=1))
smoothed_means_em = smoothRes_em["xnN"]
smoothed_covs_em = smoothRes_em["PnN"]
smoothed_std_x_y_em = np.sqrt(np.diagonal(a=smoothed_covs_em, axis1=0, axis2=1))
color_true = "blue"
color_measured = "black"
color_finite_diff = "blue"
color_ga_pattern = "rgba(255,0,0,{:f})"
color_em_pattern = "rgba(255,165,0,{:f})"
cb_alpha = 0.3

#%%
# Gradient ascent
# ~~~~~~~~~~~~~~~

#%%
# Plot true, measured and smoothed positions (with 95% confidence band)
# #####################################################################

measured_x = y[0, :]
measured_y = y[1, :]
finite_diff_x = None
finite_diff_y = None
smoothed_mean_x_ga = smoothed_means_ga[0, 0, :]
smoothed_mean_y_ga = smoothed_means_ga[3, 0, :]
smoothed_mean_x_em = smoothed_means_em[0, 0, :]
smoothed_mean_y_em = smoothed_means_em[3, 0, :]

smoothed_ci_x_upper_ga = smoothed_mean_x_ga + 1.96*smoothed_std_x_y_ga[:, 0]
smoothed_ci_x_lower_ga = smoothed_mean_x_ga - 1.96*smoothed_std_x_y_ga[:, 0]
smoothed_ci_y_upper_ga = smoothed_mean_y_ga + 1.96*smoothed_std_x_y_ga[:, 3]
smoothed_ci_y_lower_ga = smoothed_mean_y_ga - 1.96*smoothed_std_x_y_ga[:, 3]
smoothed_ci_x_upper_em = smoothed_mean_x_em + 1.96*smoothed_std_x_y_em[:, 0]
smoothed_ci_x_lower_em = smoothed_mean_x_em - 1.96*smoothed_std_x_y_em[:, 0]
smoothed_ci_y_upper_em = smoothed_mean_y_em + 1.96*smoothed_std_x_y_em[:, 3]
smoothed_ci_y_lower_em = smoothed_mean_y_em - 1.96*smoothed_std_x_y_em[:, 3]

fig = get_fig_kinematics_vs_time(
    time=time,
    measured_x=measured_x, measured_y=measured_y,
    finite_diff_x=finite_diff_x, finite_diff_y=finite_diff_y,
    ga_mean_x=smoothed_mean_x_ga, ga_mean_y=smoothed_mean_y_ga,
    ga_ci_x_upper=smoothed_ci_x_upper_ga,
    ga_ci_y_upper=smoothed_ci_y_upper_ga,
    ga_ci_x_lower=smoothed_ci_x_lower_ga,
    ga_ci_y_lower=smoothed_ci_y_lower_ga,
    em_mean_x=smoothed_mean_x_em, em_mean_y=smoothed_mean_y_em,
    em_ci_x_upper=smoothed_ci_x_upper_em,
    em_ci_y_upper=smoothed_ci_y_upper_em,
    em_ci_x_lower=smoothed_ci_x_lower_em,
    em_ci_y_lower=smoothed_ci_y_lower_em,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_finite_diff=color_finite_diff,
    color_ga_pattern=color_ga_pattern,
    color_em_pattern=color_em_pattern,
    xlabel="Time (sec)", ylabel="Position (pixels)")
# fig_filename_pattern = "../../figures/smoothed_pos.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and smoothed velocities (with 95% confidence band)
# ############################################################

measured_x = None
measured_y = None
finite_diff_x = np.diff(y[0, :])/dt
finite_diff_y = np.diff(y[1, :])/dt
smoothed_mean_x_ga = smoothed_means_ga[1, 0, :]
smoothed_mean_y_ga = smoothed_means_ga[4, 0, :]
smoothed_mean_x_em = smoothed_means_em[1, 0, :]
smoothed_mean_y_em = smoothed_means_em[4, 0, :]

smoothed_ci_x_upper_ga = smoothed_mean_x_ga + 1.96*smoothed_std_x_y_ga[:, 1]
smoothed_ci_x_lower_ga = smoothed_mean_x_ga - 1.96*smoothed_std_x_y_ga[:, 1]
smoothed_ci_y_upper_ga= smoothed_mean_y_ga + 1.96*smoothed_std_x_y_ga[:, 4]
smoothed_ci_y_lower_ga = smoothed_mean_y_ga - 1.96*smoothed_std_x_y_ga[:, 4]
smoothed_ci_x_upper_em = smoothed_mean_x_em + 1.96*smoothed_std_x_y_em[:, 1]
smoothed_ci_x_lower_em = smoothed_mean_x_em - 1.96*smoothed_std_x_y_em[:, 1]
smoothed_ci_y_upper_em= smoothed_mean_y_em + 1.96*smoothed_std_x_y_em[:, 4]
smoothed_ci_y_lower_em = smoothed_mean_y_em - 1.96*smoothed_std_x_y_em[:, 4]

fig = get_fig_kinematics_vs_time(
    time=time,
    measured_x=measured_x, measured_y=measured_y,
    finite_diff_x=finite_diff_x, finite_diff_y=finite_diff_y,
    ga_mean_x=smoothed_mean_x_ga, ga_mean_y=smoothed_mean_y_ga,
    ga_ci_x_upper=smoothed_ci_x_upper_ga,
    ga_ci_y_upper=smoothed_ci_y_upper_ga,
    ga_ci_x_lower=smoothed_ci_x_lower_ga,
    ga_ci_y_lower=smoothed_ci_y_lower_ga,
    em_mean_x=smoothed_mean_x_em, em_mean_y=smoothed_mean_y_em,
    em_ci_x_upper=smoothed_ci_x_upper_em,
    em_ci_y_upper=smoothed_ci_y_upper_em,
    em_ci_x_lower=smoothed_ci_x_lower_em,
    em_ci_y_lower=smoothed_ci_y_lower_em,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_finite_diff=color_finite_diff,
    color_ga_pattern=color_ga_pattern,
    color_em_pattern=color_em_pattern,
    xlabel="Time (sec)", ylabel="Velocity (pixels/sec)")
# fig_filename_pattern = "../../figures/smoothed_vel.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and smoothed accelerations (with 95% confidence band)
# ###############################################################

measured_x = None
measured_y = None
finite_diff_x = np.diff(np.diff(y[0, :]))/dt**2
finite_diff_y = np.diff(np.diff(y[1, :]))/dt**2
smoothed_mean_x_ga = smoothed_means_ga[2, 0, :]
smoothed_mean_y_ga = smoothed_means_ga[5, 0, :]
smoothed_mean_x_em = smoothed_means_em[2, 0, :]
smoothed_mean_y_em = smoothed_means_em[5, 0, :]

smoothed_ci_x_upper_ga = smoothed_mean_x_ga + 1.96*smoothed_std_x_y_ga[:, 2]
smoothed_ci_x_lower_ga = smoothed_mean_x_ga - 1.96*smoothed_std_x_y_ga[:, 2]
smoothed_ci_y_upper_ga = smoothed_mean_y_ga + 1.96*smoothed_std_x_y_ga[:, 5]
smoothed_ci_y_lower_ga = smoothed_mean_y_ga - 1.96*smoothed_std_x_y_ga[:, 5]
smoothed_ci_x_upper_em = smoothed_mean_x_em + 1.96*smoothed_std_x_y_em[:, 2]
smoothed_ci_x_lower_em = smoothed_mean_x_em - 1.96*smoothed_std_x_y_em[:, 2]
smoothed_ci_y_upper_em = smoothed_mean_y_em + 1.96*smoothed_std_x_y_em[:, 5]
smoothed_ci_y_lower_em = smoothed_mean_y_em - 1.96*smoothed_std_x_y_em[:, 5]

fig = get_fig_kinematics_vs_time(
    time=time,
    measured_x=measured_x, measured_y=measured_y,
    finite_diff_x=finite_diff_x, finite_diff_y=finite_diff_y,
    ga_mean_x=smoothed_mean_x_ga, ga_mean_y=smoothed_mean_y_ga,
    ga_ci_x_upper=smoothed_ci_x_upper_ga,
    ga_ci_y_upper=smoothed_ci_y_upper_ga,
    ga_ci_x_lower=smoothed_ci_x_lower_ga,
    ga_ci_y_lower=smoothed_ci_y_lower_ga,
    em_mean_x=smoothed_mean_x_em, em_mean_y=smoothed_mean_y_em,
    em_ci_x_upper=smoothed_ci_x_upper_em,
    em_ci_y_upper=smoothed_ci_y_upper_em,
    em_ci_x_lower=smoothed_ci_x_lower_em,
    em_ci_y_lower=smoothed_ci_y_lower_em,
    cb_alpha=cb_alpha,
    color_true=color_true, color_measured=color_measured,
    color_finite_diff=color_finite_diff,
    color_ga_pattern=color_ga_pattern,
    color_em_pattern=color_em_pattern,
    xlabel="Time (sec)", ylabel="Acceleration (pixels/sec^2)")
# fig_filename_pattern = "../../figures/smoothed_acc.{:s}"
# fig.write_image(fig_filename_pattern.format("png"))
# fig.write_html(fig_filename_pattern.format("html"))
fig

