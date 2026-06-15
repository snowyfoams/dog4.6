"""Top-level locomotion controller: glue between gait, convex-MPC and swing legs.

Follows the user's hopper convention: a callable `__call__(d) -> tau[12]`, with the
MPC re-planned every `replan_every` control steps and the latest ground-reaction
forces held in between. A live velocity command (set by the keyboard) shapes the
MPC reference and the swing footstep targets.
"""
import numpy as np
import mujoco

from .robot import RobotModel, Rz
from .gait import TrotGait
from .contact_gait import ContactAwareGait, MujocoFootContact, LowPassVelocity
from .convex_mpc import ConvexMPC
from .swing import SwingController


class Command:
    """Body-frame velocity command set by the user."""
    def __init__(self):
        self.vx = 0.0          # forward  (m/s)
        self.vy = 0.0          # lateral  (m/s)
        self.wz = 0.0          # yaw rate (rad/s)

    def clear(self):
        self.vx = self.vy = self.wz = 0.0


class LocomotionController:
    def __init__(self, model, replan_every=8, stand=False):
        self.m = model
        self.rob = RobotModel(model)
        self.gait = TrotGait()                   # fixed clock-driven schedule
        self.gait.stand = stand
        self.cgait = ContactAwareGait(self.gait)  # early-touchdown promotion layer
        self.touch = MujocoFootContact(model)     # sim contact estimator (swap on hw)
        self.vlp = LowPassVelocity(0.10)          # Raibert placement-velocity filter
        self.mpc = ConvexMPC(self.rob)
        self.swing = SwingController(self.rob, self.gait)
        self.cmd = Command()
        self.dt = model.opt.timestep
        self.replan_every = replan_every
        self._since = 10 ** 9
        self.f_mpc = np.zeros((4, 3))
        self.t = 0.0
        self.yaw_des = 0.0                       # absolute heading setpoint (heading hold)
        self.p_des = np.zeros(2)                 # absolute xy setpoint (station keeping)
        self.KP_POS = 1.0                        # station-keeping gain (1/s)
        self.V_CORR_MAX = 0.15                   # cap on the position correction (m/s)
        self.h_des = self.rob.nominal_h
        self.z_ref_fn = None                     # optional time-varying (z, vz) ref (jump)
        self.solve_ok = True
        self.solve_times = []

    # ------------------------------------------------------------ reference
    def _x_ref(self, com, v_cmd_world, wz):
        # xy anchored to the integrated setpoint p_des, NOT the current CoM:
        # re-anchoring on the CoM gives zero station-keeping feedback, so an
        # in-place trot random-walks away. p_des integrates the commanded
        # velocity (anti-windup clamped in __call__) and holds when cmd = 0.
        N, dt = self.mpc.N, self.mpc.dt
        ref = np.zeros((N, 13))
        for k in range(N):
            yk = self.yaw_des + wz * dt * (k + 1)        # absolute heading
            pxy = self.p_des + v_cmd_world[:2] * dt * (k + 1)
            zk, vzk = (self.h_des, 0.0) if self.z_ref_fn is None \
                else self.z_ref_fn(self.t + dt * (k + 1))
            ref[k, 0:3] = [0.0, 0.0, yk]                 # level body
            ref[k, 3:6] = [pxy[0], pxy[1], zk]
            ref[k, 6:9] = [0.0, 0.0, wz]
            ref[k, 9:12] = [v_cmd_world[0], v_cmd_world[1], vzk]
            ref[k, 12] = self.rob.g
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

        x0 = np.concatenate([rpy, com, omega, v, [self.rob.g]])
        # reconcile the clock schedule with measured contact: late-swing early
        # touchdowns are promoted to stance (else they inject unmodeled impulses
        # at gait frequency and the in-place trot pumps itself over).
        contact_now, changed = self.cgait.update(self.t, self.touch.measure(d))
        if changed:
            self._since = self.replan_every             # contact set changed -> replan
        v_place = self.vlp.update(v, self.dt)           # filtered placement velocity
        self.yaw_des += self.cmd.wz * self.dt           # integrate heading command
        self.p_des += v_cmd_world[:2] * self.dt         # integrate position setpoint
        perr = self.p_des - com[:2]                     # anti-windup: stay near the body
        pn = float(np.linalg.norm(perr))
        if pn > 0.10:
            self.p_des = com[:2] + perr * (0.10 / pn)
            perr = self.p_des - com[:2]
        # station-keeping outer loop: position error -> bounded velocity correction,
        # fed to BOTH the MPC reference and the Raibert touchdown targets. Without
        # it the swing legs keep placing feet under the drifting body and an
        # in-place trot random-walks away.
        v_corr = np.clip(self.KP_POS * perr, -self.V_CORR_MAX, self.V_CORR_MAX)
        v_eff = v_cmd_world.copy()
        v_eff[:2] += v_corr

        if self._since >= self.replan_every:
            t0 = time_now()
            x_ref = self._x_ref(com, v_eff, self.cmd.wz)
            sched = self.cgait.contact_schedule(self.t, self.mpc.N, self.mpc.dt)
            foot = rob.foot_pos(d)
            self.f_mpc, self.solve_ok = self.mpc.solve(x0, x_ref, sched, foot, com, yaw)
            self.solve_times.append(time_now() - t0)
            self._since = 0
        self._since += 1

        tau = rob.bias_tau(d)                                   # gravity/Coriolis comp
        tau += rob.grf_to_tau(d, self.f_mpc, contact_now)       # stance MPC forces
        tau += self.swing.torque(d, self.t, contact_now, v_eff, v_place)       # swing
        tau = np.clip(tau, -self.m.actuator_ctrlrange[:, 1], self.m.actuator_ctrlrange[:, 1])

        self.t += self.dt
        return tau

    def reset(self, d):
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        mujoco.mj_forward(self.m, d)
        self.t = 0.0
        self._since = 10 ** 9
        self.f_mpc[:] = 0.0
        self.yaw_des = float(self.rob.yaw(d))
        self.p_des = self.rob.com(d)[:2].copy()
        self.cgait.reset()
        self.vlp.reset()
        self.swing.was_stance[:] = True


def time_now():
    import time
    return time.perf_counter()
