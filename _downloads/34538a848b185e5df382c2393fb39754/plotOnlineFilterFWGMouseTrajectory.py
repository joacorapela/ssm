"""
Online filtering of a foraging mouse trajectory
===============================================

The code below performs online Kalman filtering of a trajectory of the mouse
shown on the left image below, as it forages in the arena shown on the right
image below. Click on the images to see their larger versions.

.. image:: /_static/mouseOnWheel.png
   :width: 300
   :alt: image of mouse on wheel

.. image:: /_static/foragingMouse.png
   :width: 300
   :alt: image of foraging mouse

"""

#%%
# Import packages
# ~~~~~~~~~~~~~~~

import sys
import os
import random
import pickle
import configparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ssm.tracking.utils
import ssm.inference

#%%
# Setup configuration variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_position = 0
number_positions = 10000
color_measures = "black"
color_pattern_filtered = "rgba(255,0,0,{:f})"
cb_alpha = 0.3
data_filename = "http://www.gatsby.ucl.ac.uk/~rapela/svGPFA/data/positions_session003_start0.00_end15548.27.csv"

#%%
# Get the mouse position measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = pd.read_csv(data_filename)
data = data.iloc[start_position:start_position+number_positions,:]
y = np.transpose(data[["x", "y"]].to_numpy())

#%%
# Get the Kalman filter configuration parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pos_x0 = y[0, 0]
pos_y0 = y[0, 0]
vel_x0 = 0.0
vel_y0 = 0.0
acc_x0 = 0.0
acc_y0 = 0.0
sigma_a = 1e4
sigma_x = 1e2
sigma_y = 1e2
sqrt_diag_V0_value = 1e-3

#%%
# Build the Kalman filter matrices for tracking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

date_times = pd.to_datetime(data["time"])
dt = (date_times.iloc[1]-date_times.iloc[0]).total_seconds()
B, Q, Qe, Z, R = ssm.tracking.utils.getLDSmatricesForKinematics_np(
    dt=dt, sigma_a=sigma_a, pos_x_R_std=sigma_x, pos_y_R_std=sigma_y)
m0 = np.array([[y[0, 0], 0, 0, y[1, 0], 0, 0]], dtype=np.double).T
V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)

#%%
# Apply the Kalman filter to the mouse position measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

onlineKF = ssm.inference.OnlineKalmanFilter(B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
filtered_means = np.empty((6, 1, y.shape[1]), dtype=np.double)
filtered_covs = np.empty((6, 6, y.shape[1]), dtype=np.double)
for i in range(y.shape[1]):
    _, _ = onlineKF.predict()
    filtered_means[:, :, i], filtered_covs[:, :, i] = \
        onlineKF.update(y=y[:, i])

#%%
# Plot the filtered positions with their 95% confidence bands
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

time = (pd.to_datetime(data["time"]) - pd.to_datetime(data["time"][0])).dt.total_seconds().to_numpy()
filter_mean_x = filtered_means[0, 0, :]
filter_mean_y = filtered_means[3, 0, :]
filter_std_x_y = np.sqrt(np.diagonal(a=filtered_covs, axis1=0, axis2=1))

filter_ci_x_upper = filter_mean_x + 1.96*filter_std_x_y[:, 0]
filter_ci_x_lower = filter_mean_x - 1.96*filter_std_x_y[:, 0]
filter_ci_y_upper = filter_mean_y + 1.96*filter_std_x_y[:, 3]
filter_ci_y_lower = filter_mean_y - 1.96*filter_std_x_y[:, 3]

fig = go.Figure()
trace_mes_x = go.Scatter(
                x=time, y=y[0, :],
                mode="markers",
                marker={"color": color_measures},
                name="measured x",
                showlegend=True,
            )
trace_mes_y = go.Scatter(
                x=time, y=y[1, :],
                mode="markers",
                marker={"color": color_measures},
                name="measured y",
                showlegend=True,
            )
trace_filter_x = go.Scatter(
                x=time, y=filter_mean_x,
                mode="markers",
                marker={"color": color_pattern_filtered.format(1.0)},
                name="filtered x",
                showlegend=True,
                legendgroup="filtered_x",
            )
trace_filter_x_cb = go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([filter_ci_x_upper,
                                  filter_ci_x_lower[::-1]]),
                fill="toself",
                fillcolor=color_pattern_filtered.format(cb_alpha),
                line=dict(color=color_pattern_filtered.format(0.0)),
                showlegend=False,
                legendgroup="filtered_x",
            )
trace_filter_y = go.Scatter(
                x=time, y=filter_mean_y,
                mode="markers",
                marker={"color":
                        color_pattern_filtered.format(1.0)},
                name="filtered y",
                showlegend=True,
                legendgroup="filtered_y",
            )
trace_filter_y_cb = go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([filter_ci_y_upper,
                                  filter_ci_y_lower[::-1]]),
                fill="toself",
                fillcolor=color_pattern_filtered.format(cb_alpha),
                line=dict(color=color_pattern_filtered.format(0.0)),
                showlegend=False,
                legendgroup="filtered_y",
            )
fig.add_trace(trace_mes_x)
fig.add_trace(trace_mes_y)
fig.add_trace(trace_filter_x)
fig.add_trace(trace_filter_x_cb)
fig.add_trace(trace_filter_y)
fig.add_trace(trace_filter_y_cb)

fig.update_layout(xaxis_title="time (seconds)", yaxis_title="position (pixels)",
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                  yaxis_range=[filter_mean_x.min(), filter_mean_x.max()])
fig
