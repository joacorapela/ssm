
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webcolors

import ssm.neural_latents.utils


def plot_latents(means, covs, bins_centers, Z=None,
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
