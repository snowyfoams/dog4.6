"""ETH-lineage locomotion controller: SLQ-MPC high level + QP-WBC low level.

Drop-in replacement for the MIT-lineage LocomotionController (same interface:
callable(d) -> tau[12], reset(d), .cmd, .gait). Everything outside the two
swapped layers is SHARED with the MIT stack — gait clock, contact-aware layer,
Raibert/Bezier swing trajectories, reference generation, station keeping,
state source — the fairness backbone of the dissertation's comparison (Ch.5
sec 5.6): only {convex QP vs SLQ} and {J^T leg controller vs QP-WBC} differ.
"""
import numpy as np
import mujoco

from .robot import RobotModel, Rz
from .gait import TrotGait
from .contact_gait import ContactAwareGait, MujocoFootContact, LowPassVelocity
from .slq_mpc import SLQMPC
from .convex_mpc import ConvexMPC
from .qp_wbc import QPWBC
from .swing import SwingController
from .controller import Command


class LocomotionControllerETH:
    """ETH stack by default; `high_level`/`low_level` flags expose the RQ4
    cross-combinations (convexMPC+WBC, SLQ+J^T) against the same shared
    backbone. The classic MIT stack lives in controller.py, untouched."""

    def __init__(self, model, replan_every=8, stand=False,
                 high_level="slq", low_level="wbc"):
        assert high_level in ("slq", "convex") and low_level in ("wbc", "jt")
        self.high_level, self.low_level = high_level, low_level
        self.m = model
        self.rob = RobotModel(model)
        self.gait = TrotGait()                    # fixed clock-driven schedule (shared)
        self.gait.stand = stand
        self.cgait = ContactAwareGait(self.gait)  # shared contact-aware layer
        self.touch = MujocoFootContact(model)
        self.vlp = LowPassVelocity(0.10)
        self.slq = SLQMPC(self.rob)               # also provides the SRB rate f(x,u)
        self.cvx = ConvexMPC(self.rob) if high_level == "convex" else None
        self.swing = SwingController(self.rob, self.gait)   # shared trajectories
        self.wbc = QPWBC(model, self.rob)
        self.cmd = Command()
        self.dt = model.opt.timestep
        self.replan_every = replan_every
        self._since = 10 ** 9
        self.t = 0.0
        self.yaw_des = 0.0
        self.p_des = np.zeros(2)
        self.KP_POS = 1.0                         # same station-keeping as MIT stack
        self.V_CORR_MAX = 0.15
        self.h_des = self.rob.nominal_h
        self.z_ref_fn = None                      # optional time-varying (z, vz) ref (jump)
        self.solve_ok = True
        # benchmark reads this: SLQ keeps its own list; convex mode is timed here
        self.solve_times = self.slq.solve_times if high_level == "slq" else []
        self.wbc_fallbacks = 0
        self._f_cvx = np.zeros((4, 3))
        self._x0_cvx = np.zeros(12)

    # ------------------------------------------------------------ reference
    def _x_ref(self, v_cmd_world, wz):
        N, dtm = self.slq.N, self.slq.dt
        ref = np.zeros((N, 12))
        for k in range(N):
            yk = self.yaw_des + wz * dtm * (k + 1)
            pxy = self.p_des + v_cmd_world[:2] * dtm * (k + 1)
            zk, vzk = (self.h_des, 0.0) if self.z_ref_fn is None \
                else self.z_ref_fn(self.t + dtm * (k + 1))
            ref[k, 0:3] = [0.0, 0.0, yk]
            ref[k, 3:6] = [pxy[0], pxy[1], zk]
            ref[k, 6:9] = [0.0, 0.0, wz]
            ref[k, 9:12] = [v_cmd_world[0], v_cmd_world[1], vzk]
        return ref

    # ------------------------------------------------------------ step
    def __call__(self, d):
        rob = self.rob
        com = rob.com(d)
        v = rob.com_vel(d)
        rpy = rob.rpy(d)
        omega = rob.omega_world(d)
        yaw = rpy[2]
        v_cmd_world = Rz(yaw) @ np.array([self.cmd.vx, self.cmd.vy, 0.0])

        contact_now, changed = self.cgait.update(self.t, self.touch.measure(d))
        if changed:
            self._since = self.replan_every       # contact set changed -> replan
        v_place = self.vlp.update(v, self.dt)
        self.yaw_des += self.cmd.wz * self.dt
        self.p_des += v_cmd_world[:2] * self.dt
        perr = self.p_des - com[:2]
        pn = float(np.linalg.norm(perr))
        if pn > 0.10:
            self.p_des = com[:2] + perr * (0.10 / pn)
            perr = self.p_des - com[:2]
        v_corr = np.clip(self.KP_POS * perr, -self.V_CORR_MAX, self.V_CORR_MAX)
        v_eff = v_cmd_world.copy()
        v_eff[:2] += v_corr

        x0 = np.concatenate([rpy, com, omega, v])
        if self._since >= self.replan_every:
            x_ref = self._x_ref(v_eff, self.cmd.wz)
            sched = self.cgait.contact_schedule(self.t, self.slq.N, self.slq.dt)
            if self.high_level == "slq":
                self.solve_ok = self.slq.solve(x0, x_ref, sched, rob.foot_pos(d))
            else:                                  # cross-combo: convex MPC high level
                import time as _time
                t0 = _time.perf_counter()
                x0_13 = np.concatenate([x0, [rob.g]])
                ref13 = np.hstack([x_ref, np.full((x_ref.shape[0], 1), rob.g)])
                self._f_cvx, self.solve_ok = self.cvx.solve(
                    x0_13, ref13, sched, rob.foot_pos(d), com, yaw)
                self._x0_cvx = x0.copy()
                self.solve_times.append(_time.perf_counter() - t0)
            self._since = 0

        if self.high_level == "slq":
            # SLQ affine policy: feedforward + surviving time-varying feedback
            f_des, x_plan, xdot_plan = self.slq.policy(self._since * self.dt, x0, contact_now)
        else:
            # convex lineage policy form: hold the planned forces (no feedback)
            f_des = self._f_cvx * np.asarray(contact_now, bool)[:, None]
            x_plan = self._x0_cvx
            self.slq._foot = rob.foot_pos(d)       # SRB rate uses current feet
            xdot_plan = self.slq.f(x0, f_des.ravel())
        self._since += 1

        # shared swing trajectories (same generator as the MIT stack)
        p_sw, v_sw, swing = self.swing.targets(d, self.t, contact_now, v_eff, v_place)

        # base task accelerations: plan feedforward + PD on the ABSOLUTE command
        # references (level attitude, station/height/velocity). Tracking the plan
        # stage instead re-anchors the reference to the drifting measured state
        # every replan and the attitude error ratchets up unchecked.
        e_rpy = np.array([0.0 - rpy[0], 0.0 - rpy[1], self.yaw_des - yaw])
        e_rpy[2] = (e_rpy[2] + np.pi) % (2 * np.pi) - np.pi
        e_world = Rz(yaw) @ np.array([e_rpy[0], e_rpy[1], 0.0]) \
            + np.array([0.0, 0.0, e_rpy[2]])
        w_ref = np.array([0.0, 0.0, self.cmd.wz])
        a_ang = np.clip(xdot_plan[6:9] + self.wbc.kp_ang * e_world
                        + self.wbc.kd_ang * (w_ref - omega), -12.0, 12.0)
        z_ref, vz_ref = (self.h_des, 0.0) if self.z_ref_fn is None \
            else self.z_ref_fn(self.t)
        p_ref = np.array([self.p_des[0], self.p_des[1], z_ref])
        v_ref = np.array([v_eff[0], v_eff[1], vz_ref])
        # per-axis gains/clips: horizontal kept tight (cone-couple safety);
        # vertical needs ~1.5 g authority and strong damping for jump thrust
        # and landing absorption (touchdown vz ~ -1.5 m/s over a short stroke)
        kd = np.array([self.wbc.kd_lin, self.wbc.kd_lin, 18.0])
        a_lin = xdot_plan[9:12] + self.wbc.kp_lin * (p_ref - com) \
            + kd * (v_ref - v)
        a_lin = np.clip(a_lin, [-6.0, -6.0, -20.0], [6.0, 6.0, 20.0])

        if self.low_level == "wbc":
            tau = self.wbc.solve(d, contact_now, f_des, a_lin, a_ang)
        else:                                      # cross-combo: J^T leg controller
            tau = None
        if tau is None:                            # J^T mode, or rare QP failure
            if self.low_level == "wbc":
                self.wbc_fallbacks += 1
            tau = rob.bias_tau(d) + rob.grf_to_tau(d, f_des, contact_now)
        # swing legs: operational-space PD overlaid on the QP torques (WBIC
        # style; same shared gains as the MIT leg controller's swing PD)
        foot = rob.foot_pos(d)
        for leg in range(4):
            if not swing[leg]:
                continue
            jp = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, d, jp, None, rob.foot_site[leg])
            J = jp[:, rob.vadr[leg]]
            F = self.swing.kp * (p_sw[leg] - foot[leg]) \
                + self.swing.kd * (v_sw[leg] - jp @ d.qvel)
            tau[rob.ctrl[leg]] = J.T @ F
        tau = np.clip(tau, -self.m.actuator_ctrlrange[:, 1],
                      self.m.actuator_ctrlrange[:, 1])
        self.t += self.dt
        return tau

    def reset(self, d):
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        mujoco.mj_forward(self.m, d)
        self.t = 0.0
        self._since = 10 ** 9
        self.yaw_des = float(self.rob.yaw(d))
        self.p_des = self.rob.com(d)[:2].copy()
        self.cgait.reset()
        self.vlp.reset()
        self.swing.was_stance[:] = True
        self.slq.U[:] = 0.0
