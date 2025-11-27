# test_em_lds.py
import numpy as np
import pytest

import ssm.inference
import ssm.learning


def simulate_lds_with_offsets(T, B, u, Q, Z, a, R, m0, V0, rng):
    """
    Simulate an LDS with offsets:

        x_0 ~ N(m0, V0)
        x_t = u + B x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = a + Z x_t + v_t,       v_t ~ N(0, R)

    Returns:
        x: (M, T+1) latent states (including x_0)
        y: (P, T)   observations
    """
    M = B.shape[0]
    P = Z.shape[0]

    x = np.zeros((M, T + 1))
    y = np.zeros((P, T))

    x[:, 0] = rng.multivariate_normal(m0, V0)

    for t in range(1, T + 1):
        w_t = rng.multivariate_normal(np.zeros(M), Q)
        x[:, t] = u + B @ x[:, t - 1] + w_t
        v_t = rng.multivariate_normal(np.zeros(P), R)
        y[:, t - 1] = a + Z @ x[:, t] + v_t

    return x, y


def kalman_loglik(y, u, B, Q, m0, V0, a, Z, R):
    """Convenience wrapper to get log-likelihood on given data."""
    kf = ssm.inference.filterLDS_SS_withMissingValues_np(
        y=y, u=u, B=B, Q=Q, m0=m0, V0=V0, a=a, Z=Z, R=R
    )
    # your filter already returns total log-likelihood
    return float(kf["logLike"])


@pytest.mark.parametrize("T_train,T_test,M,P", [(400, 400, 3, 2)])
def test_em_improves_predictive_llk(T_train, T_test, M, P):
    rng = np.random.default_rng(0)

    # --- "true" parameters, only used to generate data ---
    A_rand = rng.normal(scale=0.3, size=(M, M))
    # make B_true stable
    eigvals = np.linalg.eigvals(A_rand)
    spectral_radius = max(1.1, np.max(np.abs(eigvals)))
    B_true = A_rand / spectral_radius

    u_true = rng.normal(scale=0.5, size=M)
    a_true = rng.normal(scale=0.5, size=P)
    Q_true = 0.1 * np.eye(M)
    R_true = 0.2 * np.eye(P)
    Z_true = rng.normal(scale=0.5, size=(P, M))
    m0_true = rng.normal(scale=0.5, size=M)
    V0_true = 0.5 * np.eye(M)

    # --- simulate train and test sets from the same generative model ---
    _, y_train = simulate_lds_with_offsets(
        T=T_train,
        B=B_true,
        u=u_true,
        Q=Q_true,
        Z=Z_true,
        a=a_true,
        R=R_true,
        m0=m0_true,
        V0=V0_true,
        rng=rng,
    )
    _, y_test = simulate_lds_with_offsets(
        T=T_test,
        B=B_true,
        u=u_true,
        Q=Q_true,
        Z=Z_true,
        a=a_true,
        R=R_true,
        m0=m0_true,
        V0=V0_true,
        rng=rng,
    )

    # --- bad initial guesses (deliberately far from truth) ---
    B0 = B_true + 1.0 * rng.normal(size=B_true.shape)
    u0 = u_true + 1.0 * rng.normal(size=u_true.shape)
    Z0 = Z_true + 1.0 * rng.normal(size=Z_true.shape)
    a0 = a_true + 1.0 * rng.normal(size=a_true.shape)
    Q0 = 3.0 * np.eye(M)
    R0 = 3.0 * np.eye(P)
    m0_0 = m0_true + 1.0 * rng.normal(size=m0_true.shape)
    V0_0 = 2.0 * np.eye(M)

    # --- compute test log-likelihood with initial parameters ---
    llk_test_init = kalman_loglik(
        y=y_test, u=u0, B=B0, Q=Q0, m0=m0_0, V0=V0_0, a=a0, Z=Z0, R=R0
    )

    # --- run EM on the training set ---
    em_result = ssm.learning.em_SS_LDS(
        y=y_train,
        u0=u0,
        B0=B0,
        Q0=Q0,
        a0=a0,
        Z0=Z0,
        R0=R0,
        m0_0=m0_0,
        V0_0=V0_0,
        max_iter=50,
        tol=1e-4,
        vars_to_estimate=dict(
            m0=True, V0=True,
            u=True, B=True, Q=True,
            a=True, Z=True, R=True,
        ),
        constraint_diag_R=True,
    )

    log_like_train = em_result["log_like"]

    # --- 1. check training log-likelihood is (almost) non-decreasing ---
    diffs = np.diff(log_like_train)
    assert np.all(diffs >= -1e-6), f"Training log-likelihood not monotone: diffs={diffs}"

    # --- 2. get fitted parameters ---
    u_hat = em_result["u"]
    B_hat = em_result["B"]
    Q_hat = em_result["Q"]
    a_hat = em_result["a"]
    Z_hat = em_result["Z"]
    R_hat = em_result["R"]
    m0_hat = em_result["m0"]
    V0_hat = em_result["V0"]

    # --- 3. compute test log-likelihood with fitted parameters ---
    llk_test_fit = kalman_loglik(
        y=y_test,
        u=u_hat, B=B_hat, Q=Q_hat,
        m0=m0_hat, V0=V0_hat,
        a=a_hat, Z=Z_hat, R=R_hat,
    )

    # We don’t require parameters to match truth (they’re not identifiable),
    # but the *predictive* log-likelihood on test data should improve.
    assert llk_test_fit > llk_test_init + 1.0, (
        f"Test log-likelihood did not improve enough: "
        f"init={llk_test_init}, fit={llk_test_fit}"
    )

@pytest.fixture
def small_synthetic_data():
    """
    Construct a tiny synthetic example for testing getSummaryStatsForEM.

    We DO NOT simulate from an LDS; we just construct arbitrary but
    consistent arrays and check that the function's summation logic
    matches the mathematical definitions of the sufficient stats.
    """
    rng = np.random.default_rng(0)

    M = 2   # state dimension
    P = 3   # observation dimension
    N = 4   # number of time steps

    # Smoothed state means x_t|y, for t = 1..N
    # xnN has shape (M, 1, N) in your code
    xnN = rng.normal(size=(M, 1, N))

    # Smoothed state covariances P_t|N, for t = 1..N
    PnN = np.empty((M, M, N))
    for t in range(N):
        A = rng.normal(size=(M, M))
        PnN[:, :, t] = A @ A.T + 0.1 * np.eye(M)  # SPD

    # Smoothed initial state x_0|y
    x0N = rng.normal(size=(M, 1))
    A0 = rng.normal(size=(M, M))
    V0N = A0 @ A0.T + 0.1 * np.eye(M)

    # Smoothed lag-one covariances Cov(x_t, x_{t-1}|y) for t = 1..N
    # We'll define Pnn1N[t] = Cov(x_{t+1}, x_t), but we only need indices 0..N-1:
    #   Pnn1N[:,:,0] = Cov(x_1, x_0)
    #   Pnn1N[:,:,1] = Cov(x_2, x_1), ...
    Pnn1N = np.empty((M, M, N))
    for t in range(N):
        A = rng.normal(size=(M, M))
        Pnn1N[:, :, t] = A @ A.T  # not necessarily a covariance, but any matrix is fine for this test

    # Observations y_t (P x N)
    y = rng.normal(size=(P, N))

    # Dummy arguments that getSummaryStatsForEM expects but which our fake
    # lag1CovSmootherLDS_SS will ignore.
    Z = rng.normal(size=(P, M))
    B = rng.normal(size=(M, M))
    KN = rng.normal(size=(M, P))   # doesn't matter for this test
    Jn = rng.normal(size=(M, M, N))
    J0 = rng.normal(size=(M, M))

    return dict(
        M=M, P=P, N=N,
        xnN=xnN, PnN=PnN,
        x0N=x0N, V0N=V0N,
        Pnn1N=Pnn1N,
        y=y, Z=Z, B=B, KN=KN, Jn=Jn, J0=J0
    )


def compute_expected_stats(xnN, PnN, x0N, V0N, Pnn1N, y):
    """
    Compute the sufficient statistics directly from definitions:

        For t = 1..N:
          x_t mean = xnN[:,0,t-1]
          x_t cov  = PnN[:,:,t-1]
          Cov(x_t, x_{t-1}) = Pnn1N[:,:,t-1]
          x_0 mean = x0N[:,0]
          x_0 cov  = V0N
    """
    M, _, N = xnN.shape
    P, Ny = y.shape
    assert Ny == N

    # t = 1
    x1_mean = xnN[:, 0, 0]
    x0_mean = x0N[:, 0]

    Sxx11 = np.outer(x1_mean, x1_mean) + PnN[:, :, 0]
    Sxx10 = np.outer(x1_mean, x0_mean) + Pnn1N[:, :, 0]
    Sxx00 = np.outer(x0_mean, x0_mean) + V0N

    Tx1 = x1_mean.copy()
    Tx0 = x0_mean.copy()

    y1 = y[:, 0]
    Ty1 = y1.copy()
    Tyx11 = np.outer(y1, x1_mean)
    Tyy11 = np.outer(y1, y1)

    # t = 2..N
    for t in range(1, N):
        x_t = xnN[:, 0, t]        # E[x_t]
        x_t_minus_1 = xnN[:, 0, t-1]  # E[x_{t-1}]

        Sxx11 += np.outer(x_t, x_t) + PnN[:, :, t]
        Sxx10 += np.outer(x_t, x_t_minus_1) + Pnn1N[:, :, t]
        Sxx00 += np.outer(x_t_minus_1, x_t_minus_1) + PnN[:, :, t-1]

        Tx1 += x_t
        Tx0 += x_t_minus_1

        y_t = y[:, t]
        Ty1 += y_t
        Tyx11 += np.outer(y_t, x_t)
        Tyy11 += np.outer(y_t, y_t)

    return Sxx11, Sxx10, Sxx00, Tx1, Tx0, Ty1, Tyx11, Tyy11


def test_getSummaryStatsForEM_matches_definition(monkeypatch, small_synthetic_data):
    data = small_synthetic_data
    xnN = data["xnN"]
    PnN = data["PnN"]
    x0N = data["x0N"]
    V0N = data["V0N"]
    Pnn1N = data["Pnn1N"]
    y = data["y"]
    Z = data["Z"]
    B = data["B"]
    KN = data["KN"]
    Jn = data["Jn"]
    J0 = data["J0"]

    # --- 1. Monkeypatch lag1CovSmootherLDS_SS to return our Pnn1N ---

    def fake_lag1CovSmootherLDS_SS(Z, KN, B, Pnn, Jn, J0):
        # Ignore all inputs, just return the preconstructed lag-one covariances
        return Pnn1N

    monkeypatch.setattr(ssm.learning, "lag1CovSmootherLDS_SS", fake_lag1CovSmootherLDS_SS)

    # --- 2. Call the real getSummaryStatsForEM ---

    Sxx11, Sxx10, Sxx00, Tx1, Tx0, Ty1, Tyx11, Tyy11 = ssm.learning.getSummaryStatsForEM(
        Z=Z, B=B, KN=KN, Pnn=PnN,
        xnN=xnN, PnN=PnN, x0N=x0N, V0N=V0N,
        Jn=Jn, J0=J0, y=y
    )

    # --- 3. Compute expected stats by hand ---

    (Sxx11_exp, Sxx10_exp, Sxx00_exp,
     Tx1_exp, Tx0_exp, Ty1_exp, Tyx11_exp, Tyy11_exp) = compute_expected_stats(
        xnN=xnN, PnN=PnN, x0N=x0N, V0N=V0N, Pnn1N=Pnn1N, y=y
    )

    # --- 4. Compare via allclose ---

    atol = 1e-10
    rtol = 1e-8

    assert np.allclose(Sxx11, Sxx11_exp, rtol=rtol, atol=atol)
    assert np.allclose(Sxx10, Sxx10_exp, rtol=rtol, atol=atol)
    assert np.allclose(Sxx00, Sxx00_exp, rtol=rtol, atol=atol)
    assert np.allclose(Tx1, Tx1_exp, rtol=rtol, atol=atol)
    assert np.allclose(Tx0, Tx0_exp, rtol=rtol, atol=atol)
    assert np.allclose(Ty1, Ty1_exp, rtol=rtol, atol=atol)
    assert np.allclose(Tyx11, Tyx11_exp, rtol=rtol, atol=atol)
    assert np.allclose(Tyy11, Tyy11_exp, rtol=rtol, atol=atol)

# if __name__ == "__main__":
	# test_em_recovers_lds_with_offsets(T=500, M=3, P=2)
    # test_getSummaryStatsForEM_matches_definition(monkeypatch, small_synthetic_data)
