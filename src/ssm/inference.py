
import math
import numpy as np


class OnlineKalmanFilter:
    """
    Class implementing the online Kalman filter algorithm for the following
    linear dynamical system:

    .. math::
       x_n &= B\ x_{n-1} + w_n,\ \\textrm{where}\ w_n\sim\mathcal{N}(w_n|0, Q)\, \\textrm{and}\ x_n\in\Re^M

       y_n &= Z\ x_{n-1} + v_n,\ \\textrm{where}\ v_n\sim\mathcal{N}(v_n|0, R)\, \\textrm{and}\ y_n\in\Re^N

       x_0&\in\mathcal{N}(x_0|m_0, V_0)

    Example use:

    .. code-block:: python

        online_kf = OnlineKalmanFilter(B, Q, m0, V0, Z, R)
        x_pred, P_pred = online_kf.predict()

        for y in ys:
            x_filt, P_filt = online_kf.update(y)
            x_pred, P_pred = online_kf.predict()

    A script using `OnlineKalmanFilter` for tracking the position of a mouse
    can be found `here
    <https://github.com/joacorapela/lds_python/blob/master/code/scripts/doOnlineFilterFWGMouseTrajectory.py>`_

    Note 1:
        invocation so `predict()` and `update(y)` should alternate. That is,
        each invocation to `update(y)` should be preceded by an invocation to
        `predict()`, and each invocation to `predict()` (except the first one)
        should be preceded by an invoation to `update(y)`.

    Note 2:
        observations :math:`y_n` should be sampled uniformly.
    """
    def __init__(self, B, Q, m0, V0, Z, R):
        self._B = B
        self._Q = Q
        self._m0 = m0
        self._V0 = V0
        self._Z = Z
        self._R = R

        self._x = m0
        self._P = V0

        M = len(m0)
        self.I = np.eye(M)

    def predict(self):
        """Predicts the next state.

        :return: (state, covariance): tuple containing the predicted state vector and covariance matrix.

        """
        self.x = self.B @ self.x
        self.P = self.B @ self.P @ self.B.T + self.Q
        return self.x, self.P

    def update(self, y):
        """Updates the current state and covariance.

        :param y: observation :math:`\in\Re^M`
        :return: (state, covariance): tuple containing the updated state vector and covariance matrix.

        """
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        if not np.isnan(y).any():
            pred_obs = self.Z @ self.x
            residual = y - pred_obs
            Stmp = self.Z @ self.P @ self.Z.T + self.R
            S = (Stmp + Stmp.T) / 2
            Sinv = np.linalg.inv(S)
            K = self.P @ self.Z.T @ Sinv
            self.x = self.x + K @ residual
            self.P = (self.I - K @ self.Z) @ self.P
        return self.x, self.P

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):
        self._B = B

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    @property
    def m0(self):
        return self._m0

    @m0.setter
    def m0(self, m0):
        self._m0 = m0

    @property
    def V0(self):
        return self._V0

    @V0.setter
    def V0(self, V0):
        self._V0 = V0

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z = Z

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P


class TimeVaryingOnlineKalmanFilter:
    """
    Class implementing the time-varying and online Kalman filter algorithm for the following
    linear dynamical system:

    .. math::
       x_n &= B_n\ x_{n-1} + w_n,\ \\textrm{where}\ w_n\sim\mathcal{N}(w_n|0, Q_n)\, \\textrm{and}\ x_n\in\Re^M

       y_n &= Z_n\ x_{n-1} + v_n,\ \\textrm{where}\ v_n\sim\mathcal{N}(v_n|0, R_n)\, \\textrm{and}\ y_n\in\Re^N

       x_0&\in\mathcal{N}(x_0|m_0, V_0)

    Example use:

    .. code-block:: python

        online_kf = OnlineKalmanFilter(m0, V0)
        x_pred, P_pred = online_kf.predict()

        for y in ys:
            x_filt, P_filt = online_kf.update(y)
            x_pred, P_pred = online_kf.predict()

    A script using `OnlineKalmanFilter` for tracking the position of a mouse
    can be found `here
    <https://github.com/joacorapela/lds_python/blob/master/code/scripts/doOnlineFilterFWGMouseTrajectory.py>`_

    Note 1:
        invocation so `predict()` and `update(y)` should alternate. That is,
        each invocation to `update(y)` should be preceded by an invocation to
        `predict()`, and each invocation to `predict()` (except the first one)
        should be preceded by an invoation to `update(y)`.

    Note 2:
        observations :math:`y_n` should be sampled uniformly.
    """
    def predict(self, x, P, B, Q):
        """Predicts the next state.

        :return: (state, covariance): tuple containing the predicted state vector and covariance matrix.

        """
        x = B @ x
        P = B @ P @ B.T + Q
        return x, P

    def update(self, y, x, P, Z, R):
        """Updates the current state and covariance.

        :param y: observation :math:`\in\Re^M`
        :return: (state, covariance): tuple containing the updated state vector and covariance matrix.

        """
        # if y.ndim == 1:
        #     y = np.expand_dims(y, axis=1)
        if not np.isnan(y).any():
            M = len(x)
            I = np.eye(M)
            pred_obs = Z @ x
            innovation = y - pred_obs
            Stmp = Z @ P @ Z.T + R
            S = (Stmp + Stmp.T) / 2
            Sinv = np.linalg.inv(S)
            K = P @ Z.T @ Sinv
            x = x + K @ innovation
            P = (I - K @ Z) @ P
        return x, P


def filterLDS_SS_withMissingValues_torch(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  {xnn1, Pnn1, xnn, Pnn, innov, K, Sn, logLike}: xnn1 and Pnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Pnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    import torch
    if torch.any(torch.isnan(y[:, 0])) or torch.any(torch.isnan(y[:, -1])):
        raise ValueError("The first or last observation cannot contain nan")

    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1_h = torch.empty(size=[M, 1, N], dtype=torch.double)
    Pnn1_h = torch.empty(size=[M, M, N], dtype=torch.double)
    xnn_h = torch.empty(size=[M, 1, N], dtype=torch.double)
    Pnn_h = torch.empty(size=[M, M, N], dtype=torch.double)
    innov_h = torch.empty(size=[P, 1, N], dtype=torch.double)
    Sn_h = torch.empty(size=[P, P, N], dtype=torch.double)

    # k==0
    xnn1 = B @ m0
    Pnn1 = B @ V0 @ B.T + Q
    Stmp = Z @ Pnn1 @ Z.T + R
    Sn = (Stmp + torch.transpose(Stmp, 0, 1)) / 2
    Sinv = torch.linalg.inv(Sn)
    K = Pnn1 @ Z.T @ Sinv
    innov = y[:, 0] - (Z @  xnn1).squeeze()
    xnn = xnn1 + K @ innov
    Pnn = Pnn1 - K @ Z @ Pnn1
    logLike = -N*P*math.log(2*math.pi) - torch.logdet(Sn) - \
        innov.T @ Sinv @ innov
    if torch.isnan(logLike):
        raise RuntimeError("obtained nan log likelihood")

    xnn1_h[:, :, 0] = torch.unsqueeze(xnn1, 1)
    Pnn1_h[:, :, 0] = Pnn1
    xnn_h[:, :, 0] = torch.unsqueeze(xnn, 1)
    Pnn_h[:, :, 0] = Pnn
    innov_h[:, :, 0] = torch.unsqueeze(innov, 1)
    Sn_h[:, :, 0] = Sn

    # k>1
    for k in range(1, N):
        xnn1 = B @ xnn
        Pnn1 = B @ Pnn @ B.T + Q
        if(torch.any(torch.isnan(y[:, k]))):
            xnn = xnn1
            Pnn = Pnn1
        else:
            Stmp = Z @ Pnn1 @ Z.T + R
            Sn = (Stmp + Stmp.T)/2
            Sinv = torch.linalg.inv(Sn)
            K = Pnn1 @ Z.T @ Sinv
            innov = y[:, k] - (Z @ xnn1).squeeze()
            xnn = xnn1 + K @ innov
            Pnn = Pnn1 - K @ Z @ Pnn1
        logLike = logLike-torch.logdet(Sn) -\
            innov.T @ Sinv @ innov
        if torch.isnan(logLike):
            raise ValueError("obtained nan log likelihood")
        xnn1_h[:, :, k] = torch.unsqueeze(xnn1, 1)
        Pnn1_h[:, :, k] = Pnn1
        xnn_h[:, :, k] = torch.unsqueeze(xnn, 1)
        Pnn_h[:, :, k] = Pnn
        innov_h[:, :, k] = torch.unsqueeze(innov, 1)
        Sn_h[:, :, k] = Sn
    logLike = 0.5 * logLike
    answer = {"xnn1": xnn1_h, "Pnn1": Pnn1_h, "xnn": xnn_h, "Pnn": Pnn_h,
              "innov": innov_h, "KN": K, "Sn": Sn_h, "logLike": logLike}
    return answer


def logLikeLDS_withMissingValues_torch(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    N: dimensionality of observations

    M: dimnensionality of state

    T: number of observations

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    import torch
    if torch.any(torch.isnan(y[:, 0])) or torch.any(torch.isnan(y[:, -1])):
        raise ValueError("The first or last observation cannot contain nan")

    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    # T: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    T = y.shape[1]
    N = y.shape[0]

    xnn = m0
    Pnn = V0
    logLike = 0.0
    log2Pi = torch.log(torch.tensor(2.0 * torch.pi))
    for k in range(T):
        xnn1 = B @ xnn
        Pnn1 = B @ Pnn @ B.T + Q
        # print(Pnn1)
        # Pnn1.register_hook(print)  # This will print the gradient when it's computed.

        if(torch.any(torch.isnan(y[:, k]))):
            xnn = xnn1
            Pnn = Pnn1
        else:
            Stmp = Z @ Pnn1 @ Z.T + R
            Sn = (Stmp + Stmp.T)/2
            Sinv = torch.linalg.inv(Sn)
            K = Pnn1 @ Z.T @ Sinv
            innov = y[:, k] - (Z @ xnn1).squeeze()
            xnn = xnn1 + K @ innov
            Pnn = Pnn1 - K @ Z @ Pnn1
        logLike += -0.5 * (N * log2Pi + torch.logdet(Sn) +
                           innov.T @ torch.linalg.solve(Sn, innov))
        if torch.isnan(logLike):
            raise ValueError("obtained nan log likelihood")
    return logLike


def lds_forecast(xnn, Pnn, B, Q, h):
    """
    Forecasts the mean and covariance of a Kalman filter state at horizon h.

    Args:
        xnn (numpy array or torch tensor): Filtered state mean at time n, shape (state_dim,)
        Pnn (numpy array or torch tensor): Filtered state covariance at time n, shape (state_dim, state_dim)
        B (numpy or torch matrix): State transition matrix, shape (state_dim, state_dim)
        Q (numpy or torch matrix): Process noise covariance matrix, shape (state_dim, state_dim)
        h (int): Forecast horizon

    Returns:
        x_pred (Tensor): Forecasted mean at time n+h
        P_pred (Tensor): Forecasted covariance at time n+h
    """
    x_pred = xnn
    P_pred = Pnn

    for _ in range(h):
        P_pred = B @ P_pred @ B.T + Q
        x_pred = B @ x_pred

    return x_pred, P_pred


def lds_forecast_batch(xnn, Pnn, B, Q, m0, V0, h):
    """

	Forecasts the mean and covariance of a batch of Kalman filter states at horizon h. The first forecasted sample corresponds to sample time h-1 and the last forecasted sample correspond to sample time N+h-1; thus this function return N+1 samples.

    :param: xnn: filtered state mean at time n
    :type: xnn: numpy array or torch tensor, shape (state_dim,)

    :param: Pnn: filtered state covariance at time n
    :type: Pnn: numpy array or torch tensor, shape (state_dim, state_dim)

    :param: B: state transition matrix
    :type: B: numpy or torch matrix, shape (state_dim, state_dim)

    :param: Q: process noise covariance matrix
    :type: Q: numpy or torch matrix, shape (state_dim, state_dim)

    :param: h: forecast horizon
    :type: h: int

    :return:  (x_pred, P_pred): tuple containing the forecasted mean and covariance at time n+h
    :rtype: tuple
    """

    N = xnn.shape[2]
    x_pred = np.empty(shape=(xnn.shape[0], 1, xnn.shape[2]+1), dtype=np.double)
    P_pred = np.empty(shape=(Pnn.shape[0], Pnn.shape[1], Pnn.shape[2]+1), dtype=np.double)
    x = m0
    P = V0
    for n in range(N):
        x_pred[:, 0, n], P_pred[:, :, n] = lds_forecast(
            h=h, xnn=x, Pnn=P, B=B, Q=Q)
        x = xnn[:, 0, n]
        P = Pnn[:, :, n]
    x_pred[:, 0, N], P_pred[:, :, N] = lds_forecast(h=h, xnn=x, Pnn=P, B=B, Q=Q)
    return x_pred, P_pred


def log_like_observations_given_forecasts_lds(h, y, x_pred, P_pred, Z, R):
    first_forecasted_sample = h-1
    T = y.shape[1]
    N = y.shape[0]

    # align measurements and forecasts
    y = y[:, first_forecasted_sample:]
    x_pred = x_pred[:, :, :(T-first_forecasted_sample)]
    P_pred = P_pred[:, :, :(T-first_forecasted_sample)]

    T = y.shape[1]

    # compute log likelihood
    N_log_2pi = N * np.log(2*np.pi)
    log_like = 0.0
    num_terms = 0
    for n in range(T):
        if any(np.isnan(y[:, n])):
            continue
        yn_pred = Z @ x_pred[:, 0, n]
        innov = y[:, n] - yn_pred
        S = Z @ P_pred[:, :, n] @ Z.T + R
        mahal = innov.T @ np.linalg.solve(S, innov)
        sign, log_det = np.linalg.slogdet(S)
        if sign != 1:
            raise ValueError("S is not positive definite")
        log_like += N_log_2pi + log_det + mahal
        num_terms += 1
    log_like /= -2 * num_terms
    return log_like


def log_like_observations_given_forecasts_ekf(h, y, x_pred, P_pred, Z, Zdot, R):
    first_forecasted_sample = h-1
    T = y.shape[1]
    N = y.shape[0]

    # align measurements and forecasts
    y = y[:, first_forecasted_sample:]
    x_pred = x_pred[:, :, :(T-first_forecasted_sample)]
    P_pred = P_pred[:, :, :(T-first_forecasted_sample)]

    T = y.shape[1]

    # compute log likelihood
    N_log_2pi = N * np.log(2*np.pi)
    log_like = 0.0
    num_terms = 0
    for n in range(T):
        if any(np.isnan(y[:, n])):
            continue
        yn_pred = Z(x_pred[:, 0, n]).numpy()
        S_n = (Zdot(x_pred[:, 0, n]) @ P_pred[:, :, n] @ Zdot(x_pred[:, 0, n]).T + R(x_pred[:, 0, n])).numpy()
        innov_n = y[:, n] - yn_pred
        mahal_n = innov_n.T @ np.linalg.solve(S_n, innov_n)
        sign_n, log_det_n = np.linalg.slogdet(S_n)
        if sign_n != 1:
            raise ValueError("S is not positive definite")
        log_like += N_log_2pi + log_det_n + mahal_n
        num_terms += 1
    log_like /= -2 * num_terms
    return log_like


def ekf_forecast(xnn, Pnn, B, Bdot, Q, h):
    """
    Forecasts the mean and covariance of an EKF filter state at horizon h.

    N: dimensionality of observations

    M: dimnensionality of state

    T: number of observations

    :param: xnn: filtered state mean at time n
    :type: xnn: :math:`\Re^{N\\times 1\\times T}`

    :param: Pnn: filtered state covariance at time n
    :type: Pnn: :math:`\Re^{N\\times N\\times T}`

    :param: B: state transition function
    :type: B: :math:`\Re^M\\rightarrow\Re^M`

    :param: Bdot: Jacobian of state transition function
    :type: Bdot: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: Q: state noise covariance function
    :type: Q: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: h: forecasting horizon
    :type: h: int

    :return:  (x_pred, P_pred): tuple containing the forecasted mean and covariance at time n+h
    :rtype: tuple

    """
    x_pred = xnn
    P_pred = Pnn

    for _ in range(h):
        P_pred = Bdot(x_pred) @ P_pred @ Bdot(x_pred).T + Q(x_pred)
        x_pred = B(x_pred)

    return x_pred, P_pred


def ekf_forecast_batch(xnn, Pnn, B, Bdot, Q, m0, V0, h):
    N = xnn.shape[2]
    x_pred = np.empty(shape=(xnn.shape[0], 1, xnn.shape[2]+1), dtype=np.double)
    P_pred = np.empty(shape=(Pnn.shape[0], Pnn.shape[1], Pnn.shape[2]+1), dtype=np.double)
    x = m0
    P = V0
    for n in range(N):
        x_pred[:, 0, n], P_pred[:, :, n] = ekf_forecast(
            xnn=x, Pnn=P, B=B, Bdot=Bdot, Q=Q, h=h)
        x = xnn[:, 0, n]
        P = Pnn[:, :, n]
    x_pred[:, 0, N], P_pred[:, :, N] = ekf_forecast(
        xnn=x, Pnn=P, B=B, Bdot=Bdot, Q=Q, h=h)
    return x_pred, P_pred


def filterEKF_withMissingValues_torch(y, B, Bdot, Q, m0, V0, Z, Zdot, R,
                                      Sn_reg=1e-5):
    """ Extended Kalman filter implementation of the algorithm described in
    Chapter 10 of Durbin and Koopman 2012.

    N: dimensionality of observations

    M: dimnensionality of state

    T: number of observations

    :param: y: time series to be smoothed
    :type: y: :math:`\Re^{N\\times T}`

    :param: B: state transition function
    :type: B: :math:`\Re^M\\rightarrow\Re^M`

    :param: Bdot: Jacobian of state transition function
    :type: Bdot: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: Q: state noise covariance function
    :type: Q: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: m0: initial state mean
    :type: m0: :math:`\Re^{M}`

    :param: V0: initial state covariance
    :type: V0: :math:`\Re^{M\\times M}`

    :param: Z: state to observation function
    :type: Z: :math:`\Re^M\\rightarrow\Re^N`

    :param: Zdot: Jacobian of state to observation function
    :type: Zdot: :math:`\Re^M\\rightarrow\Re^{N\\times M}`

    :param: R: observations covariance function
    :type: R: :math:`\Re^M\\rightarrow\Re^{N\\times N}`

    :return:  {xnn1, Pnn1, xnn, Pnn, innov, K, Sn, logLike}: xnn1 and Pnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Pnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    import torch
    if torch.any(torch.isnan(y[:, 0])) or torch.any(torch.isnan(y[:, -1])):
        raise ValueError("The first or last observation cannot contain nan")

    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    # T: number of observations
    # M: dim state space
    # N: dim observations
    M = m0.shape[0]
    T = y.shape[1]
    N = y.shape[0]

    xnn1_h = torch.empty(size=[M, 1, T], dtype=torch.double)
    Pnn1_h = torch.empty(size=[M, M, T], dtype=torch.double)
    xnn_h = torch.empty(size=[M, 1, T], dtype=torch.double)
    Pnn_h = torch.empty(size=[M, M, T], dtype=torch.double)
    innov_h = torch.empty(size=[N, 1, T], dtype=torch.double)
    Sn_h = torch.empty(size=[N, N, T], dtype=torch.double)

    xnn = m0
    Pnn = V0
    logLike = 0.0
    log2Pi = torch.log(torch.tensor(2.0 * torch.pi))

    for k in range(T):
        xnn1 = B(xnn)
        Pnn1 = Bdot(xnn) @ Pnn @ Bdot(xnn).T + Q(xnn)
        if(torch.any(torch.isnan(y[:, k]))):
            xnn = xnn1
            Pnn = Pnn1
        else:
            Stmp = Zdot(xnn1) @ Pnn1 @ Zdot(xnn1).T + R(xnn1)
            Sn = (Stmp + Stmp.T)/2 + Sn_reg * torch.eye(N)
            Sinv = torch.linalg.inv(Sn)
            K = Pnn1 @ Zdot(xnn1).T @ Sinv
            innov = y[:, k] - Z(xnn1)
            xnn = xnn1 + K @ innov
            Pnn = Pnn1 - K @ Zdot(xnn1) @ Pnn1
        logLike += -0.5 * (N * log2Pi + torch.logdet(Sn) +
                           innov.T @ torch.linalg.solve(Sn, innov))
        if torch.isnan(logLike):
            raise ValueError("obtained nan log likelihood")
        xnn1_h[:, :, k] = torch.unsqueeze(xnn1, 1)
        Pnn1_h[:, :, k] = Pnn1
        xnn_h[:, :, k] = torch.unsqueeze(xnn, 1)
        Pnn_h[:, :, k] = Pnn
        innov_h[:, :, k] = torch.unsqueeze(innov, 1)
        Sn_h[:, :, k] = Sn
    answer = {"xnn1": xnn1_h, "Pnn1": Pnn1_h, "xnn": xnn_h, "Pnn": Pnn_h,
              "innov": innov_h, "KN": K, "Sn": Sn_h, "logLike": logLike}
    return answer


def logLikeEKF_withMissingValues_torch(y, B, Bdot, Q, m0, V0, Z, Zdot, R,
                                       Sn_reg=1e-5):
    """ Calculation of the log-likelihood for the extended Kalman filter model.

    N: dimensionality of observations

    M: dimnensionality of state

    T: number of observations

    :param: y: time series to be smoothed
    :type: y: :math:`\Re^{N\\times T}`

    :param: B: state transition function
    :type: B: :math:`\Re^M\\rightarrow\Re^M`

    :param: Bdot: Jacobian of state transition function
    :type: Bdot: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: Q: state noise covariance function
    :type: Q: :math:`\Re^M\\rightarrow\Re^{M\\times M}`

    :param: m0: initial state mean
    :type: m0: :math:`\Re^{M}`

    :param: V0: initial state covariance
    :type: V0: :math:`\Re^{M\\times M}`

    :param: Z: state to observation function
    :type: Z: :math:`\Re^M\\rightarrow\Re^N`

    :param: Zdot: Jacobian of state to observation function
    :type: Zdot: :math:`\Re^M\\rightarrow\Re^{N\\times M}`

    :param: R: observations covariance function
    :type: R: :math:`\Re^M\\rightarrow\Re^{N\\times N}`

    :return:  {xnn1, Pnn1, xnn, Pnn, innov, K, Sn, logLike}: xnn1 and Pnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Pnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    import torch
    if torch.any(torch.isnan(y[:, 0])) or torch.any(torch.isnan(y[:, -1])):
        raise ValueError("The first or last observation cannot contain nan")

    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    # T: number of observations
    # M: dim state space
    # N: dim observations
    M = m0.shape[0]
    T = y.shape[1]
    N = y.shape[0]

    xnn = m0
    Pnn = V0
    logLike = 0.0
    log2Pi = torch.log(torch.tensor(2.0 * torch.pi))
    for k in range(T):
        xnn1 = B(xnn)
        Pnn1 = Bdot(xnn) @ Pnn @ Bdot(xnn).T + Q(xnn)
        if(torch.any(torch.isnan(y[:, k]))):
            xnn = xnn1
            Pnn = Pnn1
        else:
            Stmp = Zdot(xnn1) @ Pnn1 @ Zdot(xnn1).T + R(xnn1)
            Sn = (Stmp + Stmp.T)/2 + Sn_reg * torch.eye(N)
            Sinv = torch.linalg.inv(Sn)
            K = Pnn1 @ Zdot(xnn1).T @ Sinv
            innov = y[:, k] - Z(xnn1)
            xnn = xnn1 + K @ innov
            Pnn = Pnn1 - K @ Zdot(xnn1) @ Pnn1
        logLike += -0.5 * (N * log2Pi + torch.logdet(Sn) +
                           innov.T @ torch.linalg.solve(Sn, innov))
        if torch.isnan(logLike):
            raise ValueError("obtained nan log likelihood")
    return logLike


def filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: u: state offset
    :type: u: numpy array (Mx1)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: a: observations offset
    :type: a: numpy array (Nx1)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  {xnn1, Pnn1, xnn, Pnn, innov, K, Sn, logLike}: xnn1 and Pnn1 (predicted means, Mx1xT, and covariances, MxMxT), xnn and Pnn (filtered means, Mx1xT, and covariances, MxMxT), innov (innovations, Nx1xT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Pnn1 = np.empty(shape=[M, M, N])
    xnn = np.empty(shape=[M, 1, N])
    Pnn = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn = np.empty(shape=[P, P, N])

    # k==0
    xnn1[:, 0, 0] = u + B @ m0
    Pnn1[:, :, 0] = B @ V0 @ B.T + Q
    Stmp = Z @ Pnn1[:, :, 0] @ Z.T + R
    Sn[:, :, 0] = (Stmp + Stmp.T) / 2
    Sinv = np.linalg.inv(Sn[:, :, 0])
    K = Pnn1[:, :, 0] @ Z.T @ Sinv
    pred_obs = a + (Z @  xnn1[:, :, 0]).squeeze()
    innov[:, 0, 0] = y[:, 0] - pred_obs
    xnn[:, :, 0] = xnn1[:, :, 0] + K @ innov[:, :, 0]
    Pnn[:, :, 0] = Pnn1[:, :, 0] - K @ Z @ Pnn1[:, :, 0]
    logLike = -N*P*np.log(2*np.pi) - np.linalg.slogdet(Sn[:, :, 0])[1] - \
        innov[:, :, 0].T @ Sinv @ innov[:, :, 0]

    # k>1
    for k in range(1, N):
        xnn1[:, 0, k] = u + (B @ xnn[:, :, k-1]).squeeze()
        Pnn1[:, :, k] = B @ Pnn[:, :, k-1] @ B.T + Q
        if(np.any(np.isnan(y[:, k]))):
            xnn[:, :, k] = xnn1[:, :, k]
            Pnn[:, :, k] = Pnn1[:, :, k]
        else:
            Stmp = Z @ Pnn1[:, :, k] @ Z.T + R
            Sn[:, :, k] = (Stmp + Stmp.T)/2
            Sinv = np.linalg.inv(Sn[:, :, k])
            K = Pnn1[:, :, k] @ Z.T @ Sinv
            pred_obs = a + (Z @  xnn1[:, :, k]).squeeze()
            innov[:, 0, k] = y[:, k] - pred_obs
            xnn[:, :, k] = xnn1[:, :, k] + K @ innov[:, :, k]
            Pnn[:, :, k] = Pnn1[:, :, k] - K @ Z @ Pnn1[:, :, k]
        logLike = logLike-np.linalg.slogdet(Sn[:, :, k])[1] -\
            innov[:, :, k].T @ Sinv @ innov[:, :, k]
    logLike = 0.5 * logLike
    answer = {"xnn1": xnn1, "Pnn1": Pnn1, "xnn": xnn, "Pnn": Pnn,
              "innov": innov, "KN": K, "Sn": Sn, "logLike": logLike}
    return answer


def smoothLDS_SS(B, xnn, Pnn, xnn1, Pnn1, m0, V0):
    """ Kalman smoother implementation

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: xnn: filtered means (from Kalman filter)
    :type: xnn: numpy array (Mx1xT)

    :param: Pnn: filtered covariances (from Kalman filter)
    :type: Pnn: numpy array (MxMXT)

    :param: xnn1: predicted means (from Kalman filter)
    :type: xnn1: numpy array (Mx1xT)

    :param: Pnn1: predicted covariances (from Kalman filter)
    :type: Pnn1: numpy array (MxMXT)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :return:  {xnN, PnN, Jn, x0N, V0N, J0}: xnn1 and Pnn1 (smoothed means, Mx1xT, and covariances, MxMxT), Jn (smoothing gain matrix, MxMxT), x0N and V0N (smoothed initial state mean, M, and covariance, MxM), J0 (initial smoothing gain matrix, MxN).

    """
    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")

    N = xnn.shape[2]
    M = B.shape[0]
    xnN = np.empty(shape=[M, 1, N])
    PnN = np.empty(shape=[M, M, N])
    Jn = np.empty(shape=[M, M, N])

    xnN[:, :, -1] = xnn[:, :, -1]
    PnN[:, :, -1] = Pnn[:, :, -1]
    for n in reversed(range(1, N)):
        Jn[:, :, n-1] = Pnn[:, :, n-1] @ B.T @ np.linalg.inv(Pnn1[:, :, n])
        xnN[:, :, n-1] = xnn[:, :, n-1] + \
            Jn[:, :, n-1] @ (xnN[:, :, n]-xnn1[:, :, n])
        PnN[:, :, n-1] = Pnn[:, :, n-1] + \
            Jn[:, :, n-1] @ (PnN[:, :, n]-Pnn1[:, :, n]) @ Jn[:, :, n-1].T
    # initial state x00 and V00
    # return the smooth estimates of the state at time 0: x0N and V0N
    J0 = V0 @ B.T @ np.linalg.inv(Pnn1[:, :, 0])
    x0N = np.expand_dims(m0, 1) + J0 @ (xnN[:, :, 0] - xnn1[:, :, 0])
    V0N = V0 + J0 @ (PnN[:, :, 0] - Pnn1[:, :, 0]) @ J0.T
    answer = {"xnN": xnN, "PnN": PnN, "Jn": Jn, "x0N": x0N, "V0N": V0N,
              "J0": J0}
    return answer
