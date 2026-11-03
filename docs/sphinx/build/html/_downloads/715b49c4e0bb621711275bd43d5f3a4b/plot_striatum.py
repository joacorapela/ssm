"""
Learning and inference of latents from striatal recordings while a mouse performed a learned poking sequence
============================================================================================================

The code below learns and infers latents from strialt recording while a mouse
performed a learned poking sequence. Recordings from [Thompson et al.,
2026](https://www.nature.com/articles/s41593-026-02362-5).

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import gcnu_common.utils.neural_data_analysis
import ssm.inference
import ssm.learning
import ssm.plotting
import ssm.neural_latents.utils
import ssm.neural_latents.plotting

#%%
# Define variables
# ^^^^^^^^^^^^^^^^

# data
bin_size = 0.2  # 200~ms
sqrt_transform = True
units_info_filename = "https://www.gatsby.ucl.ac.uk/~rapela/emmett/data/good_units_df.csv"
transition_data_filename = "https://www.gatsby.ucl.ac.uk/~rapela/emmett/data/Transition_data_sync.csv"
spikes_times_col_name = "Spike_times"

## for EJTxxx_implantx/recordingx_xx-xx-xxxx clusters 0-266 belong to the striatum
clusters_indices = np.arange(266)

# EM
n_latents = 5
n_clusters = clusters_indices.size
EM_start_time_sec = 8400.0
EM_duration_sec = 300.0
EM_max_iter = 5000
EM_tol = 1e-1
min_spike_rate_var = 0.01 # minimal spike rate variance
EM_constraint_diag_R = True
EM_min_spike_rate_var = 0.01

vars_to_estimate = {
    "u": False,
    "B": True,
    "Q": True,
    "a": True,
    "Z": True,
    "R": True,
    "m0": True,
    "V0": True,
}

# EM initial conditions
constant_u = 0.0
mean_B = 0.1
sigma_B = 0.3
mean_Q = 5.0
sigma_Q = 0.5
mean_a = 1.0
sigma_a = 0.1
mean_Z = 0.01
sigma_Z = 0.005
mean_R = 1.0
sigma_R = 0.2
constant_m0 = 0.0
mean_V0 = 0.02
sigma_V0 = 0.005

# filtering
filtering_start_time_sec = 8400.0
filtering_duration_sec = 300.0

# plotting
plotting_start_time_sec = 8400.0
plotting_duration_sec = 300.0
skip_orthonormalization = False
ports_to_plot = [1,2,3,4,5,6,7]
ports_linetypes_list = ["solid", "solid", "solid", "solid", "solid", "solid", "solid"]
ports_colors_list = ["green", "red", "cyan", "yellow", "purple", "blue", "magenta"]

ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_list))
ports_colors = dict(zip(ports_to_plot, ports_colors_list))

u0 = constant_u * np.ones(shape=(n_latents, 1))
B0 = np.random.normal(loc=mean_B, scale=sigma_B, size=(n_latents, n_latents))
a0 = np.random.normal(loc=mean_a, scale=sigma_a, size=(n_clusters, 1))
Z0 = np.random.normal(loc=mean_Z, scale=sigma_Z, size=(n_clusters, n_latents))
Q0 = np.diag(np.abs(np.random.normal(loc=mean_Q, scale=sigma_Q,
                                     size=n_latents)))
R0 = np.diag(np.abs(np.random.normal(loc=mean_R, scale=sigma_R,
                                     size=n_clusters)))
m0_0 = constant_m0 * np.ones(shape=n_latents)
V0_0 = np.diag(np.abs(np.random.normal(loc=mean_V0, scale=sigma_V0,
                                       size=n_latents)))

#%%
# Bin spikes times
# ^^^^^^^^^^^^^^^^

# read data
units_info_df = pd.read_csv(units_info_filename)

# get spikes timestamps
spikes_timestamps = [None for n in range(n_clusters)]
for cluster_index in clusters_indices:
    unit_spikes_times_str = units_info_df.iloc[cluster_index][spikes_times_col_name][1:-1].split(",")
    unit_spikes_times = np.array([float(unit_spike_times_str) for unit_spike_times_str in unit_spikes_times_str])
    spikes_timestamps[cluster_index] = unit_spikes_times

# bin spikes
t_min = min(min(st) for st in spikes_timestamps if len(st) > 0)
t_max = max(max(st) for st in spikes_timestamps if len(st) > 0)
bins_edges = np.arange(t_min, t_max, bin_size)
bins_centers = (bins_edges[1:] + bins_edges[:-1])/2

binned_spikes = np.empty(shape=(n_clusters, len(bins_centers)))
for n in range(n_clusters):
    binned_spikes[n, :] = gcnu_common.utils.neural_data_analysis.binSpikesTimes(
        spikes_times=spikes_timestamps[n], bins_edges=bins_edges,
        time_unit="sec")

if sqrt_transform:
    binned_spikes = np.sqrt(binned_spikes)

#%%
# Perform EM
# ^^^^^^^^^^

EM_data = binned_spikes.T
EM_end_time_sec = EM_start_time_sec + EM_duration_sec
valid_bins_mask = np.logical_and(EM_start_time_sec <= bins_centers,
                                 bins_centers < EM_end_time_sec)
EM_data = EM_data[valid_bins_mask, :]

spike_rate_vars = np.var(EM_data, axis=0)
valid_clusters_mask = spike_rate_vars > min_spike_rate_var
EM_data = EM_data[:, valid_clusters_mask]
valid_clusters_indices = clusters_indices[valid_clusters_mask]

a0 = a0[valid_clusters_mask]
Z0 = Z0[valid_clusters_mask,:]
R0 = R0[valid_clusters_mask, :][:, valid_clusters_mask]

# make sure that the first EM_data point is not NaN
first_not_nan_index = np.where(~np.isnan(EM_data).any(axis=1))[0][0]
EM_data = EM_data[first_not_nan_index:,]

optim_res = ssm.learning.em_SS_LDS(
    y=EM_data.T, u0=u0, B0=B0, Q0=Q0, a0=a0, Z0=Z0, R0=R0,
    m0_0=m0_0, V0_0=V0_0, max_iter=EM_max_iter, tol=EM_tol,
    vars_to_estimate=vars_to_estimate,
    constraint_diag_R=EM_constraint_diag_R,
)

#%%
# Perform filtering
# ^^^^^^^^^^^^^^^^^

filtering_data = binned_spikes.T
filtering_data = filtering_data[:, valid_clusters_mask]
filtering_end_time_sec = filtering_start_time_sec + filtering_duration_sec
filtering_valid_bins_mask = np.logical_and(
    filtering_start_time_sec <= bins_centers,
    bins_centers < filtering_end_time_sec
)
filtering_data = filtering_data[valid_bins_mask, :]
filtering_bins_centers = bins_centers[filtering_valid_bins_mask]

filtering_res = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=filtering_data.T, u=optim_res["u"], B=optim_res["B"], Q=optim_res["Q"],
    m0=optim_res["m0"], V0=optim_res["V0"], a=optim_res["a"], Z=optim_res["Z"],
    R=optim_res["R"])

#%%
# Plot filtered states
# ^^^^^^^^^^^^^^^^^^^^

# define auxiliary function
def build_events_df(start_time_sec, end_time_sec, transitions_data,
                    ports_linetypes, ports_colors,
                    start_poke_in_time_col_name="P1_IN_Ephys_TS",
                    start_port_col_name="Start_Port"):
    mask = np.logical_and(
        start_time_sec<=transitions_data[start_poke_in_time_col_name],
        transitions_data[start_poke_in_time_col_name]<end_time_sec
    )
    subset_transitions_data = transitions_data[mask]
    event_time = subset_transitions_data[start_poke_in_time_col_name]
    ports_names = subset_transitions_data[start_port_col_name]
    event_line_type = [ports_linetypes[port_name]
                       for port_name in ports_names]
    event_color = [ports_colors[port_name]
                   for port_name in ports_names]
    answer = pd.DataFrame(dict(event_time=event_time,
                               event_line_type=event_line_type,
                               event_color=event_color))
    return answer


transitions_data = pd.read_csv(transition_data_filename)

events_df = build_events_df(
    start_time_sec=filtering_start_time_sec,
    end_time_sec=filtering_end_time_sec,
    transitions_data=transitions_data,
    ports_linetypes=ports_linetypes,
    ports_colors=ports_colors)

plotting_end_time_sec = plotting_start_time_sec + plotting_duration_sec
plotting_first_index = np.where(filtering_bins_centers >= plotting_start_time_sec)[0][0]
plotting_last_index = np.where(filtering_bins_centers <= plotting_end_time_sec)[0][-1]
plotting_slice = slice(plotting_first_index, plotting_last_index)
plotting_bins_centers = filtering_bins_centers[plotting_slice]
plotting_means = filtering_res["xnn"][:,:,plotting_slice]
plotting_covs = filtering_res["Pnn"][:,:,plotting_slice]

fig = ssm.neural_latents.plotting.getPlotLatents(
    means=plotting_means,
    covs=plotting_covs,
    bins_centers=plotting_bins_centers,
    orthonormalize=not skip_orthonormalization,
    Z=optim_res["Z"],
)
ssm.plotting.add_events_vlines(fig=fig, events_df=events_df)

nro_samples = len(bins_centers)
title = ('Log-Likelihood per sample: ' f'{filtering_res["log_like"].squeeze() / nro_samples:.2E}')
fig.update_layout(title=title)
fig
