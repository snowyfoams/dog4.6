"""Swing-leg controller: Raibert footstep placement + Bezier swing arc, tracked
with an operational-space (Cartesian) PD that maps to joint torques via tau = J^T F.

Stance legs are handled by the MPC; here we only drive the legs the gait marks as
swinging. Touchdown targets follow the Raibert heuristic
    p* = p_hip_nominal + (T_st/2) v + k (v - v_cmd)
and the foot follows a cubic-in / cubic-out arc with a mid-swing apex clearance.
"""
import numpy as np
import mujoco

from .robot import Rz


class SwingController:
    def __init__(self, robot, gait, step_height=0.045, kp=500.0, kd=14.0,
                 raibert_k=0.10, foot_ground=0.016):
        self.rob = robot
        self.gait = gait
        self.step_height = step_height
        self.kp, self.kd = kp, kd
        self.raibert_k = raibert_k
        self.foot_ground = foot_ground
        self.p_liftoff = np.zeros((4, 3))               # foot pos at lift-off
        self.was_stance = np.ones(4, bool)
        self.p_des = np.zeros((4, 3))                   # last desired foot (for logging)

    @staticmethod
    def _arc(p0, p1, s, h):
        """Foot position/velocity along the swing arc, s in [0,1]."""
        # smooth horizontal interpolation (cubic, zero end-velocities)
        a = 3 * s ** 2 - 2 * s ** 3
        da = 6 * s - 6 * s ** 2
        xy = p0[:2] + a * (p1[:2] - p0[:2])
        vxy = da * (p1[:2] - p0[:2])
        # vertical: lift to apex then down (parabolic in two halves)
        z0, z1 = p0[2], p1[2]
        if s < 0.5:
            ss = s / 0.5
            z = z0 + (z0 + h - z0) * (3 * ss ** 2 - 2 * ss ** 3) - (z0) * 0  # to apex
            z = z0 + (h) * (3 * ss ** 2 - 2 * ss ** 3)
            dz = h * (6 * ss - 6 * ss ** 2) / 0.5
        else:
            ss = (s - 0.5) / 0.5
            apex = z0 + h
            z = apex + (z1 - apex) * (3 * ss ** 2 - 2 * ss ** 3)
            dz = (z1 - apex) * (6 * ss - 6 * ss ** 2) / 0.5
        return np.array([xy[0], xy[1], z]), np.array([vxy[0], vxy[1], dz])

    def targets(self, d, t, contact, v_cmd_world, v_place=None):
        """Shared swing-trajectory generator (fairness backbone, Ch.5 sec 5.6).

        Returns (p_des (4,3), v_des (4,3), swing (4,) bool). Both stacks consume
        these SAME trajectories: the MIT leg controller tracks them with a
        Cartesian PD mapped through J^T (torque() below); the ETH QP-WBC tracks
        them as task-space acceleration targets.

        v_place: low-pass-filtered CoM velocity for FOOT PLACEMENT (Raibert
        target). Raw instantaneous velocity oscillates at gait frequency and
        wobbles the touchdown targets in resonance.
        """
        rob = self.rob
        com = rob.com(d)
        v = rob.com_vel(d) if v_place is None else np.asarray(v_place)
        yaw = rob.yaw(d)
        sp = self.gait.swing_phase(t)
        foot = rob.foot_pos(d)
        p_des = foot.copy()
        v_des = np.zeros((4, 3))
        swing = np.zeros(4, bool)
        for leg in range(4):
            if contact[leg]:
                self.was_stance[leg] = True
                continue
            swing[leg] = True
            if self.was_stance[leg]:                    # just lifted off
                self.p_liftoff[leg] = foot[leg].copy()
                self.was_stance[leg] = False
            # Raibert touchdown target (xy), at ground height
            hip_xy = (com + Rz(yaw) @ rob.foot_nominal[leg])[:2]
            p_td_xy = hip_xy + 0.5 * self.gait.t_stance * v[:2] \
                + self.raibert_k * (v[:2] - v_cmd_world[:2])
            p_td = np.array([p_td_xy[0], p_td_xy[1], self.foot_ground])
            p_des[leg], v_des[leg] = self._arc(self.p_liftoff[leg], p_td,
                                               sp[leg], self.step_height)
        self.p_des = p_des
        return p_des, v_des, swing

    def torque(self, d, t, contact, v_cmd_world, v_place=None):
        """MIT leg controller: track the shared swing trajectories with a
        Cartesian PD mapped to joint torques via tau = J^T F."""
        rob = self.rob
        p_des, v_des, swing = self.targets(d, t, contact, v_cmd_world, v_place)
        foot = rob.foot_pos(d)
        tau = np.zeros(12)
        for leg in range(4):
            if not swing[leg]:
                continue
            jp = np.zeros((3, rob.m.nv))
            mujoco.mj_jacSite(rob.m, d, jp, None, rob.foot_site[leg])
            J = jp[:, rob.vadr[leg]]
            v_cur = jp @ d.qvel                          # full foot world velocity
            F = self.kp * (p_des[leg] - foot[leg]) + self.kd * (v_des[leg] - v_cur)
            tau[rob.ctrl[leg]] = J.T @ F
        return tau
