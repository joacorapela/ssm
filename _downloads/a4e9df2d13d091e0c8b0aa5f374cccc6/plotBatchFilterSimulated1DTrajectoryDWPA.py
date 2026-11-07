
"""
Offline filtering of a simulated mouse 1D trajectory
====================================================

The code below performs online Kalman filtering of a simulated mouse
trajectory.

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import numpy as np
import plotly.graph_objects as go

import ssm.kinematics.utils
import ssm.simulation
import ssm.inference

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

#%%
# Estimate velocities and accelerations using finite differences
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vel_est_fd = np.diff(y[0, :]) / dt
acc_est_fd = np.diff(vel_est_fd) / dt

#%%
# Perform batch filtering
# ~~~~~~~~~~~~~~~~~~~~~~~
# View source code of `ssm.inference.filterLDS_SS_withMissingValues_np
# <https://joacorapela.github.io/ssm/_modules/ssm/inference.html#filterLDS_SS_withMissingValues_np>`_

Q = sigma_a*Qe
filterRes = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=y, u=u, B=B, Q=Q, m0=m0, V0=V0, a=a, Z=Z, R=R)

#%%
# Set variables for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

N = y.shape[1]
time = np.arange(0, N*dt, dt)
filtered_means = filterRes["xnn"]
filtered_covs = filterRes["Pnn"]
filter_stds = np.sqrt(np.diagonal(a=filtered_covs, axis1=0, axis2=1))
color_true = "blue"
color_other = "black"
color_filtered_pattern = "rgba(255,0,0,{:f})"
cb_alpha = 0.3

#%%
# Define function for plotting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_fig_kinematics_vs_time(
        time,
        true_x,
        other_x,
        inferred_mean_x,
        inferred_ci_x_lower,
        inferred_ci_x_upper,
        cb_alpha,
        color_true,
        color_other,
        color_inferred_pattern,
        xlabel, ylabel, other_label):

    fig = go.Figure()
    trace_true_x = go.Scatter(
        x=time, y=true_x,
        mode="markers",
        marker={"color": color_true},
        name="true",
        showlegend=True,
    )
    fig.add_trace(trace_true_x)
    if other_x is not None:
        trace_other = go.Scatter(
            x=time, y=other_x,
            mode="markers",
            marker={"color": color_other},
            name=other_label,
            showlegend=True,
        )
        fig.add_trace(trace_other)
    trace_inf_x = go.Scatter(
        x=time, y=inferred_mean_x,
        mode="markers",
        marker={"color": color_inferred_pattern.format(1.0)},
        name="inferred",
        showlegend=True,
        legendgroup="inferred_x",
    )
    fig.add_trace(trace_inf_x)
    trace_inf_x_cb = go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([inferred_ci_x_upper,
                          inferred_ci_x_lower[::-1]]),
        fill="toself",
        fillcolor=color_inferred_pattern.format(cb_alpha),
        line=dict(color=color_inferred_pattern.format(0.0)),
        showlegend=False,
        legendgroup="inferred_x",
    )
    fig.add_trace(trace_inf_x_cb)

    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      # yaxis_range=[inferred_mean_x.min(),
                      #              inferred_mean_x.max()],
                     )
    return fig

#%%
# Plot true, measured and filtered positions (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[0, :]
other_x = y[0, :]
other_label = "measured"
color_other = "black"
filter_mean_x = filtered_means[0, 0, :]

filter_ci_pos_upper = filter_mean_x + 1.96*filter_stds[:, 0]
filter_ci_pos_lower = filter_mean_x - 1.96*filter_stds[:, 0]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x,
    other_x=other_x,
    inferred_mean_x=filter_mean_x,
    inferred_ci_x_upper=filter_ci_pos_upper,
    inferred_ci_x_lower=filter_ci_pos_lower,
    cb_alpha=cb_alpha,
    color_true=color_true,
    color_inferred_pattern=color_filtered_pattern,
    color_other=color_other,
    xlabel="Time (sec)", ylabel="Position (pixels)", other_label=other_label)
fig_filename_pattern = "../../../figures/filtered_pos.{:s}"
fig.write_image(fig_filename_pattern.format("png"))
fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and filtered velocities (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[1, :]
other_x = vel_est_fd
other_label = "finite diff"
color_other = "cyan"
filter_mean_x = filtered_means[1, 0, :]

filter_ci_vel_upper = filter_mean_x + 1.96*filter_stds[:, 1]
filter_ci_vel_lower = filter_mean_x - 1.96*filter_stds[:, 1]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x,
    other_x=other_x,
    inferred_mean_x=filter_mean_x,
    inferred_ci_x_upper=filter_ci_vel_upper,
    inferred_ci_x_lower=filter_ci_vel_lower,
    cb_alpha=cb_alpha,
    color_true=color_true,
    color_inferred_pattern=color_filtered_pattern,
    color_other=color_other,
    xlabel="Time (sec)", ylabel="Velocity (pixels/sec)",
    other_label=other_label)
fig_filename_pattern = "../../../figures/filtered_vel.{:s}"
fig.write_image(fig_filename_pattern.format("png"))
fig.write_html(fig_filename_pattern.format("html"))
fig

#%%
# Plot true and filtered accelerations (with 95% confidence band)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

true_x = x[2, :]
other_x = acc_est_fd
other_label = "finite diff"
color_other = "cyan"
filter_mean_x = filtered_means[2, 0, :]

filter_ci_acc_upper = filter_mean_x + 1.96*filter_stds[:, 2]
filter_ci_acc_lower = filter_mean_x - 1.96*filter_stds[:, 2]

fig = get_fig_kinematics_vs_time(
    time=time,
    true_x=true_x,
    other_x=other_x,
    inferred_mean_x=filter_mean_x,
    inferred_ci_x_upper=filter_ci_acc_upper,
    inferred_ci_x_lower=filter_ci_acc_lower,
    cb_alpha=cb_alpha,
    color_true=color_true,
    color_inferred_pattern=color_filtered_pattern,
    color_other=color_other,
    xlabel="Time (sec)", ylabel="Acceleration (pixels/sec^2)",
    other_label=other_label)
fig_filename_pattern = "../../../figures/filtered_acc.{:s}"
fig.write_image(fig_filename_pattern.format("png"))
fig.write_html(fig_filename_pattern.format("html"))
fig

