
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webcolors

import ssm.neural_latents.utils


def getPlotLatents(means, covs, bins_centers, Z=None,
                     orthogonalize=False, legend_pattern="latent {:d}",
                     cb_alpha=0.3, colors=[],
                     xlabel="Time (sec)", ylabel="Latent Value"):
    if orthogonalize:
        if Z is None:
            raise ValueError("If orthogonalize=True then Z should be "
                             "specified as a parameter")
        means, covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
            means, covs, Z)
    if len(colors) == 0:
        colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    fig = go.Figure()
    n_states = means.shape[0]
    for i in range(n_states):
        color_rgb = webcolors.hex_to_rgb(colors[i % num_colors])
        color_pattern = \
            f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},{{:f}})"
        filter_means = means[i, 0, :]
        filter_stds = np.sqrt(covs[i, i, :])
        filter_ci_upper = filter_means + 1.96*filter_stds
        filter_ci_lower = filter_means - 1.96*filter_stds

        trace = go.Scatter(
            x=bins_centers, y=filter_means,
            mode="lines+markers",
            marker={"color": color_pattern.format(1.0)},
            name=legend_pattern.format(i),
            showlegend=True,
            legendgroup=legend_pattern.format(i),
        )
        trace_cb = go.Scatter(
            x=np.concatenate([bins_centers, bins_centers[::-1]]),
            y=np.concatenate([filter_ci_upper, filter_ci_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern.format(cb_alpha),
            line=dict(color=color_pattern.format(0.0)),
            showlegend=False,
            legendgroup=legend_pattern.format(i),
        )
        fig.add_trace(trace)
        fig.add_trace(trace_cb)

    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig

def getPlotUnitsFiringRates(times, preIntensityMeans, preIntensitySTDs,
                                    cbAlpha=0.2,
                                    default_trial_color_pattern="rgba(128,128,128,{:f})",
                                    cb_transparency=0.3, mean_transparency=1.0,
                                    events_names=None,
                                    marked_events_times=None,
                                    marked_events_colors=None,
                                    marked_events_markers=None,
                                    marked_size=10,
                                    xlabel="Time (sec)",
                                    ylabel="Value",
                                    title=""):
    # times = times.detach().numpy()
    # preIntensityMeans = preIntensityMeans.detach().numpy()
    # preIntensitySTDs = preIntensitySTDs.detach().numpy()

    n_trials = preIntensityMeans.shape[0]
    if trials_ids is None:
        trials_ids = np.arange(n_trials)
    # pio.renderers.default = "browser"
    fig = go.Figure()
    for r in range(n_trials):
        trial_times = times[r, :, 0]
        meanToPlot = preIntensityMeans[r, :]
        stdToPlot = preIntensitySTDs[r, :]
        ciToPlot = 1.96*stdToPlot
        if trials_colors_patterns is not None:
            trial_color_pattern = trials_colors_patterns[r]
        else:
            trial_color_pattern = default_trial_color_pattern

        # pdb.set_trace()
#         import matplotlib
#         matplotlib.use('TkAgg')
#         import matplotlib.pyplot as plt
#         plt.plot(times, meanToPlot)
#         plt.show()
#         pdb.set_trace()

        x = trial_times
        y = meanToPlot
        y_upper = y + ciToPlot
        y_lower = y - ciToPlot

        traceCB = go.Scatter(
            x=np.concatenate((x, x[::-1])),
            y=np.concatenate((y_upper, y_lower[::-1])),
            fill="toself",
            fillcolor=trial_color_pattern.format(cb_transparency),
            line=dict(color=trial_color_pattern.format(0.0)),
            showlegend=False,
            # name="trial CB {:d}".format(r),
            legendgroup="trial{:02d}".format(trials_ids[r])
        )
        traceMean = go.Scatter(
            x=x,
            y=y,
            line=dict(color=trial_color_pattern.format(mean_transparency)),
            mode="lines",
            name="trial {:d}".format(r),
            legendgroup="trial{:02d}".format(trials_ids[r]),
            showlegend=True,
        )
        fig.add_trace(traceCB)
        fig.add_trace(traceMean)

        # add markers to trials
        if events_names is not None and\
           marked_events_times is not None and \
           marked_events_colors is not None and \
           marked_events_markers is not None and \
           align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = marked_events_times[r]-align_event_times[r]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        times[r, :, 0]-marked_events_times_centered[i]))
                    trace_marker = go.Scatter(
                        x=[times[r, marked_index, 0]],
                        y=[meanToPlot[marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        mode="markers",
                        text=[events_names[i]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "event=%{text}",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title_text=title)
    return fig


