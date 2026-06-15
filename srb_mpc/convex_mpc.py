"""Convex model-predictive controller on the single-rigid-body (SRB) model.

Di Carlo, Wensing, Katz, Bledt, Kim, "Dynamic Locomotion in the MIT Cheetah 3
Through Convex Model-Predictive Control" (IROS 2018).

State (13):  x = [theta(rpy), p_com, omega, v_com, g]   (g augmented => affine gravity)
Decision:    stacked ground-reaction forces f_i for the 4 feet over the horizon.
Dynamics linearised about the current yaw; inertia taken constant in the body frame.
The condensed problem is a dense QP solved with OSQP, with a friction pyramid and a
per-foot z-force box whose upper bound is 0 on swing feet (from the gait schedule).
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import osqp

from .robot import Rz


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


class ConvexMPC:
    def __init__(self, robot, N=10, dt=0.03, mu=0.6, fz_min=0.0, fz_max=180.0,
                 w_state=(120, 115, 140, 4, 5, 350, 4, 3, 6, 10, 11, 12),
                 w_force=2e-4):
        self.rob = robot
        self.N, self.dt = N, dt
        self.mu, self.fz_min, self.fz_max = mu, fz_min, fz_max
        self.m = robot.mass
        self.I_body = robot.I_body
        self.Iinv_body = np.linalg.inv(robot.I_body)
        # cost weights
        q = np.array(w_state, float)
        self.Qx = np.concatenate([q, [0.0]])           # 13 (gravity weight 0)
        self.Rf = w_force
        self.nf = 12                                   # 4 feet x 3
        self.prev_U = np.zeros(self.nf * N)            # warm start
        self._prob = None

    # ------------------------------------------------------------ dynamics
    def _continuous(self, yaw, foot_r):
        """Build A_c (13x13), B_c (13x12) about current yaw and foot vectors r_i (world)."""
        A = np.zeros((13, 13))
        B = np.zeros((13, 12))
        Rzy = Rz(yaw)
        A[0:3, 6:9] = Rzy.T                            # theta_dot = Rz^T omega
        A[3:6, 9:12] = np.eye(3)                       # p_dot = v
        A[11, 12] = -1.0                               # v_z_dot += -g  (x[12]=+g)
        I_world = Rzy @ self.I_body @ Rzy.T
        Iinv = np.linalg.inv(I_world)
        for i in range(4):
            B[6:9, 3 * i:3 * i + 3] = Iinv @ skew(foot_r[i])   # omega_dot
            B[9:12, 3 * i:3 * i + 3] = np.eye(3) / self.m       # v_dot
        return A, B

    def _discretize(self, A, B):
        n = 13 + self.nf
        M = np.zeros((n, n))
        M[:13, :13] = A
        M[:13, 13:] = B
        Md = sla.expm(M * self.dt)
        return Md[:13, :13], Md[:13, 13:]

    # ------------------------------------------------------------ solve
    def solve(self, x0, x_ref, contact, foot_pos, com, yaw):
        """x0 (13,), x_ref (N,13), contact (N,4) bool, foot_pos (4,3) world, com (3,)."""
        N, nf = self.N, self.nf
        foot_r = foot_pos - com
        Ad, Bd = self._discretize(*self._continuous(yaw, foot_r))

        # condensed: X = Aqp x0 + Bqp U,  X in R^{13N}, U in R^{nf N}
        Aqp = np.zeros((13 * N, 13))
        Bqp = np.zeros((13 * N, nf * N))
        Apow = np.eye(13)
        Apows = [np.eye(13)]
        for k in range(N):
            Apow = Ad @ Apow
            Apows.append(Apow)
            Aqp[13 * k:13 * k + 13] = Apow
        for k in range(N):
            for j in range(k + 1):
                Bqp[13 * k:13 * k + 13, nf * j:nf * j + nf] = Apows[k - j] @ Bd

        Qbar = np.diag(np.tile(self.Qx, N))
        Rbar = np.eye(nf * N) * self.Rf
        xref = x_ref.reshape(-1)
        H = Bqp.T @ Qbar @ Bqp + Rbar
        H = 0.5 * (H + H.T)
        g = Bqp.T @ Qbar @ (Aqp @ x0 - xref)

        # constraints: per foot per knot -> friction pyramid + fz box
        rows = []
        lb = []
        ub = []
        big = 1e6
        for k in range(N):
            for i in range(4):
                base = nf * k + 3 * i
                cmax = self.fz_max if contact[k, i] else 0.0
                cmin = self.fz_min if contact[k, i] else 0.0
                # fx - mu fz <= 0 ; -fx - mu fz <= 0  (and same for fy)
                for sgn in (+1, -1):
                    r = np.zeros(nf * N); r[base + 0] = sgn; r[base + 2] = -self.mu
                    rows.append(r); lb.append(-big); ub.append(0.0)
                    r = np.zeros(nf * N); r[base + 1] = sgn; r[base + 2] = -self.mu
                    rows.append(r); lb.append(-big); ub.append(0.0)
                # fz box
                r = np.zeros(nf * N); r[base + 2] = 1.0
                rows.append(r); lb.append(cmin); ub.append(cmax)
        C = np.array(rows)
        lb = np.array(lb); ub = np.array(ub)

        P = sp.csc_matrix(H)
        A_c = sp.csc_matrix(C)
        prob = osqp.OSQP()
        prob.setup(P=P, q=g, A=A_c, l=lb, u=ub, verbose=False,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polish=False)
        prob.warm_start(x=self.prev_U)
        res = prob.solve()
        if res.info.status_val not in (1, 2):          # solved / solved_inaccurate
            return np.zeros((4, 3)), False
        U = res.x
        self.prev_U = U
        f0 = U[:nf].reshape(4, 3)                       # first step, per-foot world force
        return f0, True
