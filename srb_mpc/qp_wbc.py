"""QP whole-body controller (ETH lineage low level), per Ch.5 eq. (wbc-qp).

At every low-level tick it solves, over z = [qdd (nv), f (12)]:

    min  || J_task qdd + Jdot qd - a_des ||^2_W  +  || f - f_des ||^2_Qf
         + w_reg ||qdd||^2
    s.t. M[:6] qdd + h[:6] = (Jc^T f)[:6]          floating-base dynamics
         Jc qdd + Jcdot qd = 0                     no-slip (stance feet)
         f_swing = 0
         |f_xy| <= mu f_z,  0 <= f_z <= f_max      friction pyramid
         tau_min <= M[6:] qdd + h[6:] - (Jc^T f)[6:] <= tau_max

and returns tau from the actuated dynamics rows. The friction cone and torque
limits live INSIDE the QP — the structural difference from the MIT leg
controller's open J^T mapping. M, h and the Jacobians come from MuJoCo
(the full rigid-body model is both the WBC's model and the simulator's truth).
"""
import numpy as np
import scipy.sparse as sp
import mujoco
import osqp


class QPWBC:
    def __init__(self, model, robot, mu=0.6, fz_min=0.0, fz_max=180.0, tau_max=18.0,
                 w_ang=50.0, w_lin=50.0, w_force=5.0, w_reg=1e-3,
                 w_contact=100.0, w_jdamp=0.1,
                 kp_lin=25.0, kd_lin=8.0, kp_ang=50.0, kd_ang=10.0,
                 kd_contact=10.0, kd_jnt=15.0):
        self.m = model
        self.rob = robot
        self.nv = model.nv                      # 18 = 6 base + 12 joints
        self.mu, self.fz_min, self.fz_max = mu, fz_min, fz_max
        self.tau_max = tau_max
        self.w = dict(ang=w_ang, lin=w_lin, f=w_force, reg=w_reg,
                      c=w_contact, jd=w_jdamp)
        self.kp_lin, self.kd_lin = kp_lin, kd_lin
        self.kp_ang, self.kd_ang = kp_ang, kd_ang
        self.kd_contact = kd_contact            # Baumgarte damping on stance feet
        self.kd_jnt = kd_jnt                    # joint-velocity damping task (stance)
        self.trunk = robot.trunk
        self._scratch = mujoco.MjData(model)    # for Jdot via integrated-q FK
        self._zprev = None
        self.fail_count = 0
        self.solve_count = 0
        # dof order follows XML document order, NOT actuator (joint-number) order;
        # map the actuated dynamics rows to d.ctrl order.
        self._act_dof = np.array([
            model.jnt_dofadr[model.actuator_trnid[a, 0]] - 6 for a in range(model.nu)])

    # ------------------------------------------------------------- jacobians
    def _jacs(self, d):
        """Base (at the SRB point) + 4 foot point-jacobians, and their Jdot*qd."""
        m, rob = self.m, self.rob
        nv = self.nv
        point = rob.com(d)

        def all_jacs(dd, pt):
            Jl = np.zeros((3, nv)); Ja = np.zeros((3, nv))
            mujoco.mj_jac(m, dd, Jl, Ja, pt, self.trunk)
            Jf = np.zeros((12, nv))
            for i, s in enumerate(rob.foot_site):
                jp = np.zeros((3, nv))
                mujoco.mj_jacSite(m, dd, jp, None, s)
                Jf[3 * i:3 * i + 3] = jp
            return Jl, Ja, Jf

        return all_jacs(d, point)

    # ------------------------------------------------------------------ solve
    def solve(self, d, contact, f_des, a_lin_des, a_ang_des):
        """Stance/base whole-body QP. Swing legs are NOT tasks here (WBIC-style):
        the controller overlays operational-space PD torques on the swing
        joints. Returns tau (12,) in actuator order, or None if the QP fails."""
        m, rob, nv = self.m, self.rob, self.nv
        nz = nv + 12
        contact = np.asarray(contact, bool)

        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy()
        Jl, Ja, Jf = self._jacs(d)

        # ----- task rows: T z = a_des  (weighted least squares)
        # J*qdot terms (Jdot qd) are deliberately DROPPED from all task targets:
        # at a 2 ms tick they are second-order, and at touchdown the just-landed
        # (fast-moving) feet carry huge Jdot qd values that the QP can only
        # cancel by dumping the base — the classic WBIC simplification.
        rows, des, wts = [], [], []
        rows.append(np.hstack([Ja, np.zeros((3, 12))]))
        des.append(a_ang_des); wts += [self.w["ang"]] * 3
        rows.append(np.hstack([Jl, np.zeros((3, 12))]))
        des.append(a_lin_des); wts += [self.w["lin"]] * 3
        # stance no-slip as a HIGH-WEIGHT task with TANGENTIAL Baumgarte damping.
        # A hard equality chatters against the simulator's soft contact; damping
        # the NORMAL component fights the touchdown impact (feet land with
        # downward velocity, the task then demands upward foot acceleration and
        # the QP satisfies it by dropping the base). So: damp x,y only.
        for leg in range(4):
            if not contact[leg]:
                continue
            Jleg = Jf[3 * leg:3 * leg + 3]
            vfoot = Jleg @ d.qvel
            target = np.zeros(3)
            target[:2] = -self.kd_contact * vfoot[:2]
            rows.append(np.hstack([Jleg, np.zeros((3, 12))]))
            des.append(target)
            wts += [self.w["c"]] * 3
        # joint-velocity damping task on STANCE-leg joints only (continuity;
        # the pure min-norm solution chatters). Swing joints are torque-overlaid
        # by the controller and must not be damped here.
        for leg in range(4):
            if not contact[leg]:
                continue
            cols = rob.vadr[leg]
            Jj = np.zeros((3, nz))
            Jj[np.arange(3), cols] = 1.0
            rows.append(Jj); des.append(-self.kd_jnt * d.qvel[cols])
            wts += [self.w["jd"]] * 3
        # force tracking rows: f - f_des
        Ff = np.hstack([np.zeros((12, nv)), np.eye(12)])
        rows.append(Ff); des.append(np.asarray(f_des, float).ravel())
        wts += [self.w["f"]] * 12

        T = np.vstack(rows)
        a = np.concatenate(des)
        W = np.asarray(wts, float)
        P = T.T @ (T * W[:, None])
        P[np.arange(nv), np.arange(nv)] += self.w["reg"]
        qvec = -T.T @ (W * a)

        # ----- constraints  A z in [l, u]
        A_list, lo, up = [], [], []
        # floating-base dynamics rows (unactuated): M[:6] qdd - Jc^T[:6] f = -h[:6]
        A_list.append(np.hstack([M[:6], -Jf[:, :6].T]))
        lo.append(-h[:6]); up.append(-h[:6])
        # (no-slip is handled as a high-weight task in the cost above)
        # swing forces pinned to zero; stance friction pyramid
        for leg in range(4):
            S = np.zeros((6, nz))
            c = nv + 3 * leg
            S[0, c] = 1; S[0, c + 2] = -self.mu      # fx - mu fz <= 0
            S[1, c] = -1; S[1, c + 2] = -self.mu     # -fx - mu fz <= 0
            S[2, c + 1] = 1; S[2, c + 2] = -self.mu
            S[3, c + 1] = -1; S[3, c + 2] = -self.mu
            S[4, c + 2] = 1                          # 0 <= fz <= fmax
            S[5, c] = 0
            A_list.append(S[:5])
            if contact[leg]:
                # fz_min keeps a little normal load on scheduled stance feet so
                # tangential friction capacity exists through load transients
                lo += [-np.inf * np.ones(4), np.array([self.fz_min])]
                up += [np.zeros(4), np.array([self.fz_max])]
            else:                                    # pin f = 0
                Z = np.zeros((3, nz))
                Z[0, c] = Z[1, c + 1] = Z[2, c + 2] = 1
                A_list[-1] = Z
                lo += [np.zeros(3)]; up += [np.zeros(3)]
        # torque limits on the actuated rows
        A_list.append(np.hstack([M[6:], -Jf[:, 6:].T]))
        lo.append(-self.tau_max - h[6:]); up.append(self.tau_max - h[6:])

        Amat = np.vstack(A_list)
        l = np.concatenate(lo); u = np.concatenate(up)

        prob = osqp.OSQP()
        prob.setup(P=sp.csc_matrix(P), q=qvec, A=sp.csc_matrix(Amat), l=l, u=u,
                   verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=4000,
                   polish=False)
        if self._zprev is not None and self._zprev.size == nz:
            prob.warm_start(x=self._zprev)
        res = prob.solve()
        self.solve_count += 1
        if res.info.status_val not in (1, 2):        # solved / solved inaccurate
            self.fail_count += 1
            return None
        z = res.x
        self._zprev = z.copy()
        qdd, f = z[:nv], z[nv:]
        tau_dof = M[6:] @ qdd + h[6:] - Jf[:, 6:].T @ f   # actuated rows, dof order
        tau = tau_dof[self._act_dof]                      # -> actuator (d.ctrl) order
        return np.clip(tau, -self.tau_max, self.tau_max)
