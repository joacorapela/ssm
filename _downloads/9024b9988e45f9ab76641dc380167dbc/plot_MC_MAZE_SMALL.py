


"""
Learning and inference of latents with MC_MAZE_SMALL data
============================================================

The code below learns and infers latents with MC_MAZE_SMALL data.

"""

import numpy as np
import plotly.graph_objects as go

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import ssm.inference
import ssm.learning
import ssm.neural_latents.utils
import ssm.neural_latents.plotting


#%%
# Define variables
# ^^^^^^^^^^^^^^^^
# Here I am setting a small value to variable ``max_iter``, in order to build
# this documentation quickly. You may want to set this variable to
# ``max_iter=2000`` in combination with a small error tolerance ``tol=1e-3``.
#
# In addition, you may want to plot all the data, and not just a small time
# interval between ``from_time = 100.0`` and ``to_time = 130.0``, by setting
# ``from_time = -np.inf`` and ``to_time = np.inf``.

# data
get_data_from_Dandi = True
dandiset_ID = "000140"
dandi_filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
local_filepath = f"../../../../projects/lds_neuralLatents_MC_MAZE_SMALL/data/{dandiset_ID}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
bin_size = 0.02

# plot
events_names = ["start_time", "target_on_time", "go_cue_time",
                "move_onset_time", "stop_time"]
events_linetypes = ["dot", "dash", "dashdot", "longdash", "solid"]
events_colors_spikes = ["white", "white", "white", "white", "white"]
events_colors_latents = ["black", "black", "black", "black", "black"]
cb_alpha = 0.3
from_time = 100.0
to_time = 130.0

# model
n_latents = 10

# estimation initial conditions
sigma_B = 0.1
sigma_Z = 0.1
sigma_Q = 0.1
sigma_R = 0.1
sigma_m0 = 0.1
sigma_V0 = 0.1

# estimation parameters
# max_iter = 2000
max_iter = 2000
tol = 1e-1
vars_to_estimate = {"B": True, "Q": True, "Z": True, "R": True,
                    "m0": True, "V0": True, }

#%%
# Download data
# ^^^^^^^^^^^^^
if get_data_from_Dandi:
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_ID,
                                    "draft").get_asset_by_path(dandi_filepath)
        s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
        io = NWBHDF5IO(s3_path, mode="r", driver="ros3")
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.intervals["trials"].to_dataframe()
else:
    with NWBHDF5IO(local_filepath, 'r') as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.intervals["trials"].to_dataframe()


# n_clusters
n_clusters = units_df.shape[0]

#%%
# Bin spikes
# ^^^^^^^^^^

# continuous spikes times
continuous_spikes_times = [None for n in range(n_clusters)]
for n in range(n_clusters):
    continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

binned_spikes, bin_edges = ssm.neural_latents.utils.bin_spike_times(
    spike_times=continuous_spikes_times, bin_size=bin_size)
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
transformed_binned_spikes = np.sqrt(binned_spikes + 0.5)

# clip data to plot
first_index = np.where(bin_centers >= from_time)[0][0]
last_index = np.where(bin_centers <= to_time)[0][-1]
to_plot_slice = slice(first_index, last_index)
bin_centers_to_plot = bin_centers[to_plot_slice]
trials_df = trials_df[np.logical_and(
    trials_df['start_time'] >= from_time,
    trials_df['stop_time'] <= to_time,
)]

#%%
# Plot binned spikes
# ~~~~~~~~~~~~~~~~~~

fig = go.Figure()
trace = go.Heatmap(x=bin_centers_to_plot,
                   z=transformed_binned_spikes[:, to_plot_slice],
                   colorbar=dict(title="<b>Sqrt(spike_count+0.5)</b>"))
fig.add_trace(trace)
ssm.neural_latents.plotting.add_events_vlines(
    fig=fig, trials_df=trials_df, events_names=events_names,
    events_linetypes=events_linetypes, events_colors=events_colors_spikes)
fig.update_xaxes(title="Time (sec)")
fig.update_yaxes(title="Cluster Index")
fig

#%%
# Parameter learning using expectation maximization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B0 = np.diag(np.random.normal(loc=0, scale=sigma_B, size=n_latents))
Z0 = np.random.normal(loc=0, scale=sigma_Z, size=(n_clusters, n_latents))
Q0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_Q, size=n_latents)))
R0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_R, size=n_clusters)))
m0_0 = np.random.normal(loc=0, scale=sigma_m0, size=n_latents)
V0_0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_V0, size=n_latents)))

optim_res = ssm.learning.em_SS_LDS(
    y=transformed_binned_spikes, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
    m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
    vars_to_estimate=vars_to_estimate,
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
# Kalman filtering
# ^^^^^^^^^^^^^^^^

filter_res = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=transformed_binned_spikes, B=optim_res["B"], Q=optim_res["Q"],
    m0=optim_res["m0"], V0=optim_res["V0"], Z=optim_res["Z"], R=optim_res["R"])

#%%
# Kalman smoothing
# ^^^^^^^^^^^^^^^^

smoothing_res = ssm.inference.smoothLDS_SS(
    B=optim_res["B"], xnn=filter_res["xnn"], Pnn=filter_res["Pnn"],
    xnn1=filter_res["xnn1"], Pnn1=filter_res["Pnn1"],
    m0=optim_res["m0"], V0=optim_res["V0"])

#%%
# Plot smoothed states
# ^^^^^^^^^^^^^^^^^^^^

o_means_to_plot, o_covs_to_plot = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
    means=smoothing_res["xnN"][:, :, to_plot_slice],
    covs=smoothing_res["PnN"][:, :, to_plot_slice], Z=optim_res["Z"])

fig = ssm.neural_latents.plotting.plot_latents(
    means=o_means_to_plot,
    covs=o_covs_to_plot,
    bin_centers=bin_centers_to_plot,
    trials_df=trials_df,
    events_names=events_names,
    events_linetypes=events_linetypes,
    events_colors=events_colors_latents,
    cb_alpha=cb_alpha,
    legend_pattern="smoothing_{:d}",
)

fig
