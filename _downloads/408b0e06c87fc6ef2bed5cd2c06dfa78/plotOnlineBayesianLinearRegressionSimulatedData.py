
"""
Simulated data
==============

The code below uses an online algorithm to estimate the posterior of the weighs
of a linear regression model using simulate data.

"""

#%%
# Import requirments
# ------------------

import numpy as np
import scipy.stats
import plotly.subplots
import plotly.graph_objects as go

import ssm.inference

#%%
# Define data generation variables
# --------------------------------

n_samples = 20
a0 = -0.3
a1 = 0.5
likelihood_precision_coef = (1/0.2)**2
n_samples_to_plot = (1, 2, 20)

#%%
# Generate data
# -------------

x = np.random.uniform(low=-1, high=1, size=n_samples)
y = a0 + a1 * x
t = y + np.random.standard_normal(size=y.shape) * 1.0/likelihood_precision_coef

#%%
# Define plotting variables
# -------------------------
n_post_samples = 6
marker_true = "cross"
size_true = 10
color_true = "white"
marker_data = "circle-open"
size_data = 10
color_data = "blue"
line_width_data = 5


#%%
# Define estimation variables
# ---------------------------

prior_precision_coef = 2.0

#%%
# Build Kalman filter matrices
# ----------------------------
B = np.eye(N=2)
Q = np.zeros(shape=((2,2)))
R = np.array([[1.0/likelihood_precision_coef]])

#%%
# Estimate and plot posterior
# ---------------------------

x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_grid, Y_grid))

Phi = np.column_stack((np.ones(len(x)), x))

# set prior
m0 = np.array([0.0, 0.0])
S0 = 1.0 / prior_precision_coef * np.eye(2)

fig = plotly.subplots.make_subplots(rows=len(n_samples_to_plot)+1, cols=3)
x_dense = np.arange(-1.0, 1.0, 0.1)

# trace true coefficient
trace_true_coef = go.Scatter(x=[a0], y=[a1], mode="markers",
                             marker_symbol=marker_true,
                             marker_size=size_true,
                             marker_color=color_true,
                             name="true mean",
                             showlegend=False)

rv = scipy.stats.multivariate_normal(m0, S0)

# plot prior
Z = rv.pdf(pos)
trace_post = go.Contour(x=x_grid, y=y_grid, z=Z, showscale=False)
fig.add_trace(trace_post, row=1, col=2)

fig.add_trace(trace_true_coef, row=1, col=2)

fig.update_xaxes(title_text="Intercept", row=1, col=2)
fig.update_yaxes(title_text="Slope", row=1, col=2)
# sample from prior
samples = rv.rvs(size=n_post_samples)

# plot regression lines corresponding to samples
for a_sample in samples:
    sample_intercept, sample_slope = a_sample
    sample_y = sample_intercept + sample_slope * x_dense
    trace = go.Scatter(x=x_dense, y=sample_y, mode="lines",
                       line_color="red", showlegend=False)
    fig.add_trace(trace, row=1, col=3)
fig.update_xaxes(title_text="x", row=1, col=3)
fig.update_yaxes(title_text="y", row=1, col=3)

mn = m0
Sn = S0
kf = ssm.inference.TimeVaryingOnlineKalmanFilter()

for n, t in enumerate(y):
    print(f"Processing {n}/({len(y)})")
    # update posterior
    mn, Sn = kf.predict(x=mn, P=Sn, B=B, Q=Q)
    mn, Sn = kf.update(y=t, x=mn, P=Sn, Z=Phi[n, :].reshape((1, Phi.shape[1])), R=R)

    if n+1 in n_samples_to_plot:
        index_sample = n_samples_to_plot.index(n+1)
        # compute likelihood
        Z = np.empty(shape=(len(x_grid), len(y_grid)), dtype=np.double)
        for i, w0 in enumerate(x_grid):
            for j, w1 in enumerate(y_grid):
                rv = scipy.stats.norm(w0 + w1 * x[n],
                                      1.0/likelihood_precision_coef)
                Z[j, i] = rv.pdf(t)

        # plot likelihood
        trace_like = go.Contour(x=x_grid, y=y_grid, z=Z, showscale=False)
        fig.add_trace(trace_like, row=index_sample+2, col=1)

        fig.add_trace(trace_true_coef, row=index_sample+2, col=1)

        fig.update_xaxes(title_text="Intercept", row=index_sample+2, col=1)
        fig.update_yaxes(title_text="Slope", row=index_sample+2, col=1)

        rv = scipy.stats.multivariate_normal(mn, Sn)

        # plot updated posterior
        Z = rv.pdf(pos)
        trace_post = go.Contour(x=x_grid, y=y_grid, z=Z, showscale=False)
        fig.add_trace(trace_post, row=index_sample+2, col=2)

        fig.add_trace(trace_true_coef, row=index_sample+2, col=2)

        fig.update_xaxes(title_text="Intercept", row=index_sample+2, col=2)
        fig.update_yaxes(title_text="Slope", row=index_sample+2, col=2)

        # sample from posterior
        samples = rv.rvs(size=n_post_samples)

        # plot regression lines corresponding to samples
        for a_sample in samples:
            sample_intercept, sample_slope = a_sample
            sample_y = sample_intercept + sample_slope * x_dense
            trace = go.Scatter(x=x_dense, y=sample_y, mode="lines",
                               line_color="red", showlegend=False)
            fig.add_trace(trace, row=index_sample+2, col=3)
        trace_data = go.Scatter(x=x[:(n+1)], y=y[:(n+1)],
                                mode="markers",
                                marker_symbol=marker_data,
                                marker_size=size_data,
                                marker_color=color_data,
                                marker_line_width=line_width_data,
                                showlegend=False,
                               )
        fig.add_trace(trace_data, row=index_sample+2, col=3)
        fig.update_xaxes(title_text="x", row=index_sample+2, col=3)
        fig.update_yaxes(title_text="y", row=index_sample+2, col=3)

fig
