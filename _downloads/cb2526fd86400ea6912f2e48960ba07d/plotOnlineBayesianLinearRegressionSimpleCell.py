
"""
Anesthetized cat V1 simple cell
===============================

The code below uses an online algorithm to estimate the linear receptive field
of a simple cell recorded from an anesthetized cat, as described in "Touryan,
Jon and Felsen, Gidon and Dan, Yang (2005). Spatial structure of complex cell
receptive fields measured with natural images. Neuron 45 (5), 781-791."

"""

#%%
# Import requirments
# ------------------

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

import ssm.inference

#%%
# Define variables
# ----------------
n_samples_to_use = 1000
response_delay_samples = 1
prior_precision_coef = 2.0
likelihood_precision_coef = 0.1
fig_update_delay = 0.1
images_filename = "https://www.gatsby.ucl.ac.uk/~rapela/neuroinformatics/2024/worksheets/linearRegression/data/equalpower_C2_25hzPP.dat"
responses_filename = "https://www.gatsby.ucl.ac.uk/~rapela/neuroinformatics/2024/worksheets/linearRegression/data/040909.a.c06.C2_nsSumSpikeRates.dat"

#%%
# Get data
# --------
images = pd.read_csv(images_filename, sep="\s+").to_numpy()
responses = pd.read_csv(responses_filename, sep="\s+").to_numpy().flatten()
Phi = np.column_stack((np.ones(len(images)), images))
n_pixels = images.shape[1]
image_width = int(np.sqrt(n_pixels))
image_height = image_width

Phi = Phi[:-response_delay_samples,]
responses = responses[response_delay_samples:]

n_samples_to_use = min(Phi.shape[0], n_samples_to_use)
print(f"Using {n_samples_to_use} out of {Phi.shape[0]} samples")
Phi = Phi[:n_samples_to_use,]
responses = responses[:n_samples_to_use]

#%%
# Build Kalman filter matrices
# ----------------------------
B = np.eye(N=n_pixels+1)
Q = np.zeros(shape=((n_pixels+1, n_pixels+1)))
R = np.array([[1.0/likelihood_precision_coef]])

#%%
# Estimate posterior
# ------------------

# set prior
m0 = np.zeros((n_pixels+1,), dtype=np.double)
S0 = 1.0 / prior_precision_coef * np.eye(n_pixels+1, dtype=np.double)
indices = np.arange(len(m0))

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1, adjustable="box", aspect=1)
ax2 = fig.add_subplot(2, 1, 2)

mn = m0
Sn = S0
kf = ssm.inference.TimeVaryingOnlineKalmanFilter()
for n, t in enumerate(responses):
    print(f"Processing {n}/({len(responses)})")
    # update posterior
    mn, Sn = kf.predict(x=mn, P=Sn, B=B, Q=Q)
    mn, Sn = kf.update(y=t, x=mn, P=Sn, Z=Phi[n, :].reshape((1, Phi.shape[1])), R=R)

    # plot posterior
    stds = np.sqrt(np.diag(Sn))
    ax1.clear()
    ax1.contourf(mn[1:].reshape((image_width, image_height)))
    title = (r"$\alpha={:.02f},\beta={:.02f},\lambda={:.02f},"
             "{:d}/{:d}$".format(
                 prior_precision_coef, likelihood_precision_coef,
                 prior_precision_coef/likelihood_precision_coef,
                 n, len(responses)))
    ax1.set_title(title)
    ax2.clear()
    ax2.errorbar(x=indices, y=mn, yerr=1.96*stds)
    ax2.axhline(y=0)
    ax2.set_xlabel("Pixel index")
    ax2.set_ylabel("Pixel value")
    # Note that using time.sleep does *not* work here!
    plt.pause(fig_update_delay)

