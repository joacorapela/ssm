import numpy as np


def simulateLDS(T, u, B, Q, m0, V0, a, Z, R):
    """ Simulation of linear dynamical system

    :param: T: number of observations
    :type: T: int

    :param: u: state offset
    :type: u: numpy array (Mx1)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (Mx1)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: a: observation offset
    :type: u: numpy array (Nx1)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return: (initial state, states, observations)
    :rtype: tuple(numpy.array[M], numpy.array[M, T], numpy.array[P, T])

    """
    M = B.shape[0]
    P = Z.shape[0]
    # state noise
    w = np.random.multivariate_normal(np.zeros(M), Q, T).T
    # measurement noise
    v = np.random.multivariate_normal(np.zeros(P), R, T).T
    # initial state noise
    x = np.empty(shape=(M, T))
    y = np.empty(shape=(P, T))
    x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
    x[:, 0] = u + B @ x0 + w[:, 0]
    for n in range(1, T):
        x[:, n] = u + B @ x[:, n-1] + w[:, n]
    y = a[:, np.newaxis] + Z @ x + v
    return x0, x, y


def simulateNDSgaussianNoise(T, B, Q, m0, V0, Z, R):
    """ Simulation of nonlinear dynamical system with Gaussian noise

    :param: T: number of observations
    :type: T: int

    :param: B: state transition function
    :type: B: :math:`\\Re^M\\rightarrow\\Re^M`

    :param: Q: state noise covariance function
    :type: Q: :math:`\\Re^M\\rightarrow\\Re^{M\\times M}`

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation function
    :type: Z: :math:`\\Re^M\\rightarrow\\Re^N`

    :param: R: observations covariance function
    :type: R: :math:`\\Re^M\\rightarrow\\Re^{N\\times N}`

    :return: (initial state, states, observations)
    :rtype: tuple(numpy.array[M], numpy.array[M, T], numpy.array[P, T])

    """

    M = m0.shape[0]
    x = np.empty(shape=(M, T))
    x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
    w = np.random.multivariate_normal(np.zeros(M), Q(x0), 1).flatten()
    x[:, 0] = B(x0) + w

    aux_cov = R(x[:, 0])
    N = aux_cov.shape[0]
    v = np.random.multivariate_normal(np.zeros(N), aux_cov, 1).flatten()
    y = np.empty(shape=(N, T))
    y[:, 0] = Z(x[:, 0]) + v

    for n in range(1, T):
        w_nM1 = np.random.multivariate_normal(np.zeros(M),
                                              Q(x[:, n-1]), 1).flatten()
        x[:, n] = B(x[:, n-1]) + w_nM1
        v_n = np.random.multivariate_normal(np.zeros(N),
                                            R(x[:, n]), 1).flatten()
        y[:, n] = Z(x[:, n]) + v_n

    return x0, x, y
