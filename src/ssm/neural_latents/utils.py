
import numpy as np


def ortogonalizeMeansAndCovs(means, covs, Z):
    U, S, Vh = np.linalg.svd(Z)
    SVh = np.diag(S) @ Vh
    o_means = np.einsum("ik,kjt->ijt",
                        SVh, means-np.mean(means, axis=2, keepdims=True))  # \in ijt
    # aux1 = Vh @ covs_to_plot  # \in ijl
    aux1 = np.einsum("ik,kjt->ijt", SVh, covs)  # \in ijt
    aux2 = SVh.T  # \in jk
    o_covs = np.einsum("ijt,jk->ikt", aux1, aux2)
    return o_means, o_covs


def ortogonalizeMeansAndCovsWithLoop(means, covs, Z):
    U, S, Vh = np.linalg.svd(Z)
    SVh = np.diag(S) @ Vh
    T = means.shape[2]
    o_means, o_covs = np.empty_like(means), np.empty_like(covs)
    for t in range(T):
        o_means[:, 0, t] = SVh @ means[:, 0, t]
        o_covs[:, :, t] = SVh @ covs[:, :, t] @ SVh.T
    return o_means, o_covs


def bin_spike_times(spike_times, bin_size=0.1, t_max=None):
    """
    Bin spike times into spike count matrix.

    Parameters
    ----------
    spike_times : list of lists
        spike_times[n] = list of spike times (floats) for neuron n
    bin_size : float
        Bin width in seconds (default 0.1 = 100 ms)
    t_max : float or None
        End time for binning. If None, set to max spike time across all
        neurons.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Binned spike counts.
    bin_edges : ndarray
        The bin edges used.
    """
    n_neurons = len(spike_times)
    if t_max is None:
        t_max = max(max(st) for st in spike_times if len(st) > 0)

    # Compute bin edges
    bin_edges = np.arange(0, t_max + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    # Allocate matrix
    spike_counts = np.zeros((n_neurons, n_bins), dtype=int)

    # Bin spikes per neuron
    for n, spikes in enumerate(spike_times):
        spike_counts[n], _ = np.histogram(spikes, bins=bin_edges)

    return spike_counts, bin_edges


def matrix_to_string(A):
    return ";".join([",".join(map(str, row)) for row in A])


def vector_to_string(v):
    return ",".join(map(str, v))
