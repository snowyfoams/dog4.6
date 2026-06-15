"""SLQ-MPC: iterated dynamic programming (discrete iLQR/DDP) on the NONLINEAR
single-rigid-body model — the high level of the ETH-lineage stack.

Same SRB reduction, horizon (N=10, dt=0.03 s), foot treatment and cost weights
as the convex MPC (fairness), but the orientation dynamics are kept nonlinear
(full R(rpy), Euler-rate map, gyroscopic term) and the OCP is solved by the SLQ
iteration of Ch.5 sec 5.4: forward rollout -> linearise/quadratise -> backward
Riccati sweep (Q-blocks) -> line search, warm-started so 1-2 iterations per
control tick suffice (real-time iteration). The product is the affine
time-varying policy
    u(t) = u_bar(n) + K(n) (x - x_bar(n)),
whose feedback gains K SURVIVE between replans — the distinguishing feature vs
the convex stack's feedforward-dominant forces.

State x = [rpy(3), p(3), omega_world(3), v(3)]  (level-frame attitude, as in
robot.RobotModel). Input u = 12 ground-reaction forces (4 feet x 3, world).
Swing legs are masked per stage from the contact schedule; the friction pyramid
is enforced by clamping in the rollout (the QP-WBC downstream enforces it hard).
"""
import time
import numpy as np

from .robot import Rz


def rpy_to_R(rpy):
    """ZYX (yaw about z, then pitch, then roll): R = Rz(psi) Ry(th) Rx(phi)."""
    phi, th, psi = rpy
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(th), np.sin(th)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
        [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
        [-sth,      cth * sph,                   cth * cph]])


def euler_rate_inv(rpy):
    """E^{-1}(rpy): maps world angular velocity -> ZYX Euler-angle rates."""
    phi, th, psi = rpy
    cth = max(np.cos(th), 1e-4)        # gimbal guard (|pitch| < 90 deg in practice)
    sth = np.sin(th)
    cps, sps = np.cos(psi), np.sin(psi)
    # omega = E(rpy) [phid, thd, psid] with E = [[c_th c_ps, -s_ps, 0],
    #                                            [c_th s_ps,  c_ps, 0],
    #                                            [-s_th,      0,    1]]
    return np.array([
        [cps / cth,        sps / cth,        0.0],
        [-sps,             cps,              0.0],
        [cps * sth / cth,  sps * sth / cth,  1.0]])


class SLQMPC:
    NX, NU = 12, 12

    def __init__(self, robot, N=10, dt=0.03, mu=0.6, fz_max=180.0,
                 w_state=(120, 115, 140, 4, 5, 350, 4, 3, 6, 10, 11, 12),
                 w_force=2e-4, w_cone=0.02, iters=2, reg0=1e-6):
        self.rob = robot
        self.N, self.dt = N, dt
        self.mu, self.fz_max = mu, fz_max
        self.m = robot.mass
        self.g = robot.g
        self.I_body = robot.I_body                  # level frame == world at stand
        self.Q = np.diag(np.asarray(w_state, float))
        self.R = np.eye(self.NU) * w_force
        self.w_cone = w_cone                        # smooth friction-pyramid penalty
        self.iters = iters
        self.reg0 = reg0

        self.U = np.zeros((N, self.NU))             # warm-start buffer
        self.Xbar = np.zeros((N + 1, self.NX))
        self.K = [np.zeros((self.NU, self.NX)) for _ in range(N)]
        self.solve_times = []
        self.last_cost = np.inf
        self._foot = np.zeros((4, 3))
        self._sched = np.ones((N, 4), bool)

    # ---------------------------------------------------------------- dynamics
    def clamp(self, u, active):
        """Per-leg friction-pyramid clamp; swing legs forced to zero."""
        u = u.reshape(4, 3).copy()
        u[~active] = 0.0
        fz = np.clip(u[:, 2], 0.0, self.fz_max)
        lim = self.mu * fz
        u[:, 0] = np.clip(u[:, 0], -lim, lim)
        u[:, 1] = np.clip(u[:, 1], -lim, lim)
        u[:, 2] = fz
        return u.ravel()

    def f(self, x, u):
        """Continuous nonlinear SRB dynamics xdot = f(x, u)."""
        rpy, p, w, v = x[0:3], x[3:6], x[6:9], x[9:12]
        F = u.reshape(4, 3)
        Rw = rpy_to_R(rpy)
        Iw = Rw @ self.I_body @ Rw.T
        tau = np.cross(self._foot - p, F).sum(axis=0)
        wdot = np.linalg.solve(Iw, tau - np.cross(w, Iw @ w))
        out = np.empty(12)
        out[0:3] = euler_rate_inv(rpy) @ w
        out[3:6] = v
        out[6:9] = wdot
        out[9:12] = F.sum(axis=0) / self.m
        out[11] -= self.g
        return out

    def step(self, x, u, k):
        """Discrete (Euler, dt) step; swing legs masked. The friction cone is
        NOT clamped here: a hard clamp inside the rollout is invisible to the
        linearisation and stalls the iteration at the cone boundary. It is
        handled by the smooth penalty (cone_terms) and a final output clamp."""
        ue = u.reshape(4, 3).copy()
        ue[~self._sched[k]] = 0.0
        return x + self.dt * self.f(x, ue.ravel())

    def cone_terms(self, u):
        """Quadratic penalty (cost, grad, Gauss-Newton Hessian) for friction-
        pyramid violations: |f_xy| <= mu f_z, f_z >= 0 (relaxed-barrier style)."""
        c, g, H = 0.0, np.zeros(self.NU), np.zeros((self.NU, self.NU))
        w = self.w_cone
        for leg in range(4):
            i = 3 * leg
            fx, fy, fz = u[i], u[i + 1], u[i + 2]
            if fz < 0.0:
                c += 0.5 * w * fz * fz
                g[i + 2] += w * fz
                H[i + 2, i + 2] += w
            lim = self.mu * max(fz, 0.0)
            dz = -self.mu if fz > 0.0 else 0.0
            for j, fxy in ((i, fx), (i + 1, fy)):
                v = abs(fxy) - lim
                if v > 0.0:
                    s = 1.0 if fxy >= 0 else -1.0
                    c += 0.5 * w * v * v
                    g[j] += w * v * s
                    g[i + 2] += w * v * dz
                    H[j, j] += w
                    H[j, i + 2] += w * s * dz
                    H[i + 2, j] += w * s * dz
                    H[i + 2, i + 2] += w * dz * dz
        return c, g, H

    def linearize(self, x, u, k):
        """A by central differences of the cheap numpy step(); B analytic
        (the GRFs enter linearly: vdot rows I/m, wdot rows Iw^{-1} [r-p]x)."""
        A = np.zeros((self.NX, self.NX))
        h = 1e-5
        for i in range(self.NX):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            A[:, i] = (self.step(xp, u, k) - self.step(xm, u, k)) / (2 * h)
        Rw = rpy_to_R(x[0:3])
        Iw_inv = np.linalg.inv(Rw @ self.I_body @ Rw.T)
        B = np.zeros((self.NX, self.NU))
        for leg in range(4):
            if not self._sched[k][leg]:
                continue
            r = self._foot[leg] - x[3:6]
            rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            B[6:9, 3 * leg:3 * leg + 3] = self.dt * (Iw_inv @ rx)
            B[9:12, 3 * leg:3 * leg + 3] = self.dt / self.m * np.eye(3)
        return A, B

    # ------------------------------------------------------------------- cost
    def stage_cost(self, x, u, k):
        e = x - self._xref[k]
        return (0.5 * e @ self.Q @ e + 0.5 * u @ self.R @ u
                + self.cone_terms(u)[0])

    def traj_cost(self, X, U):
        J = sum(self.stage_cost(X[k], U[k], k) for k in range(self.N))
        eN = X[self.N] - self._xref[self.N - 1]
        return J + 0.5 * eN @ self.Q @ eN          # terminal = stage weights

    def rollout(self, x0, U, Xref=None, Uref=None, K=None, alpha=1.0):
        X = np.zeros((self.N + 1, self.NX)); X[0] = x0
        Un = np.zeros_like(U)
        for k in range(self.N):
            if K is None:
                Un[k] = U[k]
            else:
                Un[k] = Uref[k] + alpha * U[k] + K[k] @ (X[k] - Xref[k])
            Un[k] = Un[k] * np.repeat(self._sched[k], 3)   # swing mask only
            X[k + 1] = X[k] + self.dt * self.f(X[k], Un[k])
        return X, Un

    # ------------------------------------------------------------------ solve
    def solve(self, x0, x_ref, sched, foot):
        """One SLQ-MPC tick (real-time iteration, warm-started).

        x0 (12,), x_ref (N,12), sched (N,4) bool, foot (4,3) world.
        Returns solve_ok. Plan is stored in self.Xbar, self.U, self.K.
        """
        t0 = time.perf_counter()
        self._xref = np.asarray(x_ref, float)
        self._sched = np.asarray(sched, bool)
        self._foot = np.asarray(foot, float)

        U = self.U.copy()
        # re-seat warm start: newly-stance legs get the static load share
        nst = max(self._sched[0].sum(), 1)
        for k in range(self.N):
            for leg in range(4):
                if self._sched[k][leg] and abs(U[k, 3 * leg + 2]) < 1e-9:
                    U[k, 3 * leg + 2] = self.m * self.g / nst
        X, U = self.rollout(x0, U)
        J = self.traj_cost(X, U)
        ok = True
        mu_reg = self.reg0
        for _ in range(self.iters):
            A = [None] * self.N; B = [None] * self.N
            for k in range(self.N):
                A[k], B[k] = self.linearize(X[k], U[k], k)
            eN = X[self.N] - self._xref[self.N - 1]
            Vx, Vxx = self.Q @ eN, self.Q.copy()
            kff = [None] * self.N; K = [None] * self.N
            for k in range(self.N - 1, -1, -1):
                e = X[k] - self._xref[k]
                _, cg, cH = self.cone_terms(U[k])
                Qx = self.Q @ e + A[k].T @ Vx
                Qu = self.R @ U[k] + cg + B[k].T @ Vx
                Qxx = self.Q + A[k].T @ Vxx @ A[k]
                Quu = self.R + cH + B[k].T @ Vxx @ B[k] + mu_reg * np.eye(self.NU)
                Qux = B[k].T @ Vxx @ A[k]
                Quu_inv = np.linalg.inv(Quu)
                kff[k] = -Quu_inv @ Qu
                K[k] = -Quu_inv @ Qux
                Vx = Qx + K[k].T @ Quu @ kff[k] + K[k].T @ Qu + Qux.T @ kff[k]
                Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
                Vxx = 0.5 * (Vxx + Vxx.T)
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1):
                Xn, Un = self.rollout(x0, np.array(kff), X, U, K, alpha)
                Jn = self.traj_cost(Xn, Un)
                if Jn < J:
                    X, U, J = Xn, Un, Jn
                    improved = True
                    break
            self.K = K
            if not improved:
                mu_reg *= 10
                if mu_reg > 1e2:
                    break
        self.Xbar, self.U, self.last_cost = X, U, J
        self.solve_times.append(time.perf_counter() - t0)
        return ok

    # ------------------------------------------------------------------ policy
    def policy(self, elapsed, x, contact):
        """Affine time-varying policy between replans: the SLQ control law.

        elapsed: seconds since the plan's t0. Returns (forces (4,3), x_plan,
        xdot_plan) — the plan state/derivative feed the WBC base task.
        """
        n = min(int(elapsed / self.dt), self.N - 1)
        u = self.U[n] + self.K[n] @ (x - self.Xbar[n])
        u = self.clamp(u, np.asarray(contact, bool))
        xp = self.Xbar[n]
        return u.reshape(4, 3), xp, self.f(xp, self.clamp(self.U[n], self._sched[n]))
