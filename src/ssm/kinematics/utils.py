import math
import torch


def getLDSmatricesForKinematics_torch(dt, sigma_a, pos_x_R_std, pos_y_R_std):
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3

    # Eq. 6.3.3-2
    import torch
    B = torch.tensor([[1, dt, .5*dt**2, 0, 0, 0],
                     [0, 1, dt, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, dt, .5*dt**2],
                     [0, 0, 0, 0, 1, dt],
                     [0, 0, 0, 0, 0, 1]],
                    dtype=torch.double)
    Z = torch.tensor([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]],
                    dtype=torch.double)
    # Eq. 6.3.3-4
    Qt = torch.tensor([[dt**4/4, dt**3/2, dt**2/2],
                      [dt**3/2, dt**2,   dt],
                      [dt**2/2, dt,      1]],
                     dtype=torch.double)
    Qe = torch.zeros(size=(6, 6), dtype=torch.double)
    Qe[:3, :3] = Qt
    Qe[3:6, 3:6] = Qt
    Q = sigma_a**2 * Qe
    R = torch.diag(torch.tensor([pos_x_R_std**2, pos_y_R_std**2]))

    return B, Q, Qe, Z, R

def getLDSmatricesForKinematics_np(dt, sigma_a, pos_x_R_std, pos_y_R_std):
    B, Q, Qe, Z, R = getLDSmatricesForKinematics_torch(dt, sigma_a, pos_x_R_std, pos_y_R_std)
    return B.numpy(), Q.numpy(), Qe.numpy(), Z.numpy(), R.numpy()

def getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt, alpha, sigma_a,
        cos_theta_Q_std, sin_theta_Q_std,
        omega_Q_std,
        pos_x_R_std, pos_y_R_std,
        cos_theta_R_std, sin_theta_R_std):
    import torch

    def getBdot(dt, alpha, epsilon=1e-10):
        def Bdot(x):
            pos_x = x[0]
            vel_x = x[1]
            acc_x = x[2]
            pos_y = x[3]
            vel_y = x[4]
            acc_y = x[5]
            cos_theta = x[6]
            sin_theta = x[7]
            omega = x[8]

            v = math.sqrt(vel_x**2 + vel_y**2 + epsilon)
            A = ((1 - alpha * v * dt) * cos_theta - omega * dt * sin_theta +
                 alpha * dt * vel_x)
            B = ((1 - alpha * v * dt) * sin_theta + omega * dt * cos_theta +
                 alpha * dt * vel_y)

            pAWRTxdot = -alpha * dt * cos_theta * vel_x / v + alpha * dt
            pBWRTxdot = -alpha * dt * sin_theta * vel_x / v
            pSpeedWRTxdot = (A * pAWRTxdot + B * pBWRTxdot) / math.sqrt(A**2 + B**2)
            pCosWRTxdot = (pAWRTxdot * math.sqrt(A**2 + B**2) -
                           A * pSpeedWRTxdot) / (A**2 + B**2)
            pSinWRTxdot = (pBWRTxdot * math.sqrt(A**2 + B**2) -
                           B * pSpeedWRTxdot) / (A**2 + B**2)

            pAWRTydot = -alpha * dt * cos_theta * vel_y / v
            pBWRTydot = -alpha * dt * sin_theta * vel_y / v + alpha * dt
            pSpeedWRTydot = (A * pAWRTydot + B * pBWRTydot) / math.sqrt(A**2 + B**2)
            pCosWRTydot = (pAWRTydot * math.sqrt(A**2 + B**2) -
                           A * pSpeedWRTydot) / (A**2 + B**2)
            pSinWRTydot = (pBWRTydot * math.sqrt(A**2 + B**2) -
                           B * pSpeedWRTydot) / (A**2 + B**2)

            pAWRTcosTheta = 1 - alpha * v * dt
            pBWRTcosTheta = omega * dt
            pSpeedWRTcosTheta = (A * pAWRTcosTheta + B * pBWRTcosTheta) / math.sqrt(A**2 + B**2)
            pCosWRTcosTheta = (pAWRTcosTheta * math.sqrt(A**2 + B**2) -
                               A * pSpeedWRTcosTheta) / (A**2 + B**2)
            pSinWRTcosTheta = (pBWRTcosTheta * math.sqrt(A**2 + B**2) -
                               B * pSpeedWRTcosTheta) / (A**2 + B**2)

            pAWRTsinTheta = -omega * dt
            pBWRTsinTheta = 1 - alpha * v * dt
            pSpeedWRTsinTheta = (A * pAWRTsinTheta + B * pBWRTsinTheta) / math.sqrt(A**2 + B**2)
            pCosWRTsinTheta = (pAWRTsinTheta * math.sqrt(A**2 + B**2) -
                               A * pSpeedWRTsinTheta) / (A**2 + B**2)
            pSinWRTsinTheta = (pBWRTsinTheta * math.sqrt(A**2 + B**2) -
                               B * pSpeedWRTsinTheta) / (A**2 + B**2)

            pAWRTomega = -dt * sin_theta
            pBWRTomega = dt * cos_theta
            pSpeedWRTomega = (A * pAWRTomega + B * pBWRTomega) / math.sqrt(A**2 + B**2)
            pCosWRTomega = (pAWRTomega * math.sqrt(A**2 + B**2) -
                            A * pSpeedWRTomega) / (A**2 + B**2)
            pSinWRTomega = (pBWRTomega * math.sqrt(A**2 + B**2) -
                            B * pSpeedWRTomega) / (A**2 + B**2)

            aBdot = torch.zeros(size=(9, 9), dtype=torch.double)
            aBdot[0, 0] = 1; aBdot[0, 1] = dt; aBdot[0, 2] = .5*dt**2
            aBdot[1, 1] = 1; aBdot[1, 2] = dt
            aBdot[2, 2] = 1
            aBdot[3, 3] = 1; aBdot[3, 4] = dt; aBdot[3, 5] = .5*dt**2
            aBdot[4, 4] = 1; aBdot[4, 5] = dt
            aBdot[5, 5] = 1
            aBdot[6, 1] = pCosWRTxdot; aBdot[6, 4] = pCosWRTydot; aBdot[6, 6] = pCosWRTcosTheta; aBdot[6, 7] = pCosWRTsinTheta; aBdot[6, 8] = pCosWRTomega
            aBdot[7, 1] = pSinWRTxdot; aBdot[7, 4] = pSinWRTydot; aBdot[7, 6] = pSinWRTcosTheta; aBdot[7, 7] = pSinWRTsinTheta; aBdot[7, 8] = pSinWRTomega
            aBdot[8, 8] = 1
            return aBdot
        return Bdot

    def getB(dt, alpha):
        def B(x):
            pos_x = x[0]
            vel_x = x[1]
            acc_x = x[2]
            pos_y = x[3]
            vel_y = x[4]
            acc_y = x[5]
            cos_theta = x[6]
            sin_theta = x[7]
            omega = x[8]

            v = math.sqrt(vel_x**2 + vel_y**2)
            answer = torch.empty(size=(9,), dtype=torch.double)
            answer[0] = pos_x + dt * vel_x + .5 * dt**2 * acc_x
            answer[1] = vel_x + dt * acc_x
            answer[2] = acc_x
            answer[3] = pos_y + dt * vel_y + .5 * dt**2 * acc_y
            answer[4] = vel_y + dt * acc_y
            answer[5] = acc_y
            A = (alpha * dt * vel_x +
                 (1 - alpha * v * dt) * cos_theta -
                 omega * dt * sin_theta)
            B = (alpha * dt * vel_y +
                 (1 - alpha * v * dt) * sin_theta +
                 omega * dt * cos_theta)
            answer[6] = A / math.sqrt(A**2 + B**2)
            answer[7] = B / math.sqrt(A**2 + B**2)
            answer[8] = omega
            return answer
        return B

    def getZdot():
        def Zdot(x):
            Zmat = torch.zeros(size=(4, 9), dtype=torch.double)
            Zmat[0, 0] = 1
            Zmat[1, 3] = 1
            Zmat[2, 6] = 1
            Zmat[3, 7] = 1
            return Zmat
        return Zdot

    def getZ():
        def Z(x):
            Zmat = getZdot()(x=None)
            answer = Zmat @ x
            return answer
        return Z

    def getQ(dt, sigma_a, cos_theta_Q_std, sin_theta_Q_std,
             omega_Q_std):
        def Q(x):
            Qt = torch.tensor([[dt**4/4, dt**3/2, dt**2/2],
                               [dt**3/2, dt**2, dt],
                               [dt**2/2, dt, 1]], dtype=torch.double)
            Qe = torch.zeros(size=(9, 9), dtype=torch.double)
            Qe[:3, :3] = Qt
            Qe[3:6, 3:6] = Qt
            e6 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              dtype=torch.double)
            e7 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                              dtype=torch.double)
            e8 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                              dtype=torch.double)
            Q = (Qe * sigma_a**2 +
                 cos_theta_Q_std**2 * torch.outer(e6, e6) +
                 sin_theta_Q_std**2 * torch.outer(e7, e7) +
                 omega_Q_std**2 * torch.outer(e8, e8))
            return Q
        return Q

    def getR(pos_x_R_std, pos_y_R_std,
             cos_theta_R_std,
             sin_theta_R_std):
        def R(x):
            e0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
            e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
            e2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
            e3 = torch.tensor([0.0, 0.0, 0.0, 1.0])

            R = (pos_x_R_std**2 * torch.outer(e0, e0) +
                 pos_y_R_std**2 * torch.outer(e1, e1) +
                 cos_theta_R_std**2 * torch.outer(e2, e2) +
                 sin_theta_R_std**2 * torch.outer(e3, e3))
            if any(torch.isnan(R.flatten()).tolist()):
                print("nan detected in R")
                breakpoint()
            return R
        return R

    B = getB(dt=dt, alpha=alpha)
    Bdot = getBdot(dt=dt, alpha=alpha)
    Z = getZ()
    Zdot = getZdot()
    Q = getQ(dt=dt, sigma_a=sigma_a,
             cos_theta_Q_std=cos_theta_Q_std,
             sin_theta_Q_std=sin_theta_Q_std,
             omega_Q_std=omega_Q_std)
    R = getR(pos_x_R_std=pos_x_R_std, pos_y_R_std=pos_y_R_std,
             cos_theta_R_std=cos_theta_R_std,
             sin_theta_R_std=sin_theta_R_std)
    return B, Bdot, Z, Zdot, Q, R
