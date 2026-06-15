"""Thin MuJoCo interface for the DOG3.0 floating-base quadruped.

Hides index bookkeeping and gives the controller exactly what the SRB-MPC and
swing controller need: CoM state, body orientation/rates, foot positions, per-leg
foot Jacobians (via mj_jacSite) and the GRF->torque map.

Canonical leg order is [FR, FL, RR, RL]; per leg the joints are [abduction, thigh, knee].
The model uses MuJoCo's numeric Jacobians, so the ~60deg export tilt and per-leg
axis-sign differences need no special handling here.
"""
import numpy as np
import mujoco

LEG_NAMES = ["FR", "FL", "RR", "RL"]
# joint NUMBER (jointN) per leg: [abduction, thigh, knee]
LEG_JOINT_NUM = {
    "FR": [12, 11, 10],
    "FL": [9, 8, 7],
    "RR": [1, 3, 2],
    "RL": [6, 5, 4],
}


class RobotModel:
    def __init__(self, model):
        self.m = model
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_site = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{l}_foot") for l in LEG_NAMES])
        # qpos / dof / ctrl indices, shape (4 legs, 3 joints)
        self.qadr = np.zeros((4, 3), int)
        self.vadr = np.zeros((4, 3), int)
        self.ctrl = np.zeros((4, 3), int)
        for li, leg in enumerate(LEG_NAMES):
            for ji, num in enumerate(LEG_JOINT_NUM[leg]):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{num}")
                self.qadr[li, ji] = model.jnt_qposadr[jid]
                self.vadr[li, ji] = model.jnt_dofadr[jid]
                self.ctrl[li, ji] = num - 1            # actuator m{num} at index num-1
        self.mass = float(model.body_mass.sum())
        self.g = float(-model.opt.gravity[2])
        self._scratch = mujoco.MjData(model)          # for gravity-comp evaluation
        # R_corr = trunk orientation at the standing keyframe. The "level" body frame
        # is R_level = R_trunk @ R_corr^T, so upright/standing => identity (rpy=0).
        d0 = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, d0, 0)
        mujoco.mj_forward(model, d0)
        self.R_corr = d0.xmat[self.trunk].reshape(3, 3).copy()
        # SRB regulation point: the trunk-FIXED point that coincides with the standing
        # composite CoM. The trunk's own CoM sits ~3 cm off the weight line (legs are
        # ~70% of the mass), which biases every moment the MPC computes and rolls the
        # trot over; the composite CoM itself jitters as the legs swing. A body-fixed
        # point at the standing composite CoM gives an unbiased, smooth SRB point.
        R0 = d0.xmat[self.trunk].reshape(3, 3)
        self._com_local = R0.T @ (d0.subtree_com[self.trunk] - d0.xpos[self.trunk])
        self.nominal_h = float(self.com(d0)[2])                   # standing SRB-point height
        self.I_body = self._composite_inertia(d0)      # 3x3 about CoM, LEVEL frame
        # nominal hip (abduction-anchor) and foot offsets from CoM in the LEVEL frame
        self.hip_body = self._hip_offsets(d0)
        self.foot_nominal = (d0.site_xpos[self.foot_site] - self.com(d0)).copy()
        # Centre the nominal foothold rectangle laterally on the SRB point: the stand
        # keyframe is hip-symmetric (matches the CAD pose) but the weight line sits
        # ~3 cm to one side of the hip centerline. Trot touchdowns must straddle the
        # WEIGHT, not the hips, or the body perpetually falls sideways and trips.
        self.foot_nominal[:, 1] = (np.sign(self.foot_nominal[:, 1])
                                   * np.abs(self.foot_nominal[:, 1]).mean())

    # ---------------------------------------------------------------- state
    # The SRB point is body-fixed (no jitter from leg swing) but placed at the
    # STANDING composite CoM, so the MPC's moment arms are unbiased. See __init__.
    def com(self, d):
        return d.xpos[self.trunk] + d.xmat[self.trunk].reshape(3, 3) @ self._com_local

    def _spatial_vel(self, d):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.trunk, res, 0)
        return res                                    # [omega(3), v(3)] world, at trunk CoM

    def com_vel(self, d):
        sv = self._spatial_vel(d)                     # anchored at trunk CoM (xipos)
        r = self.com(d) - d.xipos[self.trunk]
        return sv[3:6] + np.cross(sv[0:3], r)         # shift to the SRB point

    def quat(self, d):
        return d.xquat[self.trunk].copy()             # (w,x,y,z)

    def R(self, d):
        return d.xmat[self.trunk].reshape(3, 3).copy()

    def R_level(self, d):
        """Physical (leveled) body orientation in world: identity when standing."""
        return d.xmat[self.trunk].reshape(3, 3) @ self.R_corr.T

    def rpy(self, d):
        return mat2rpy(self.R_level(d))

    def yaw(self, d):
        return mat2rpy(self.R_level(d))[2]

    def omega_world(self, d):
        return self._spatial_vel(d)[0:3].copy()

    def foot_pos(self, d):
        return d.site_xpos[self.foot_site].copy()     # (4,3) world

    def foot_vel(self, d, leg):
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, None, self.foot_site[leg])
        return jp @ d.qvel

    def foot_jac(self, d, leg):
        """3x3 world-frame foot Jacobian wrt that leg's [abd, thigh, knee] dofs."""
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, None, self.foot_site[leg])
        return jp[:, self.vadr[leg]]

    # ---------------------------------------------------------------- torque map
    def grf_to_tau(self, d, forces, contact):
        """tau[12] from per-leg world GRFs. tau_leg = -J^T f for stance legs."""
        tau = np.zeros(12)
        for leg in range(4):
            if not contact[leg]:
                continue
            J = self.foot_jac(d, leg)
            tau[self.ctrl[leg]] = -J.T @ forces[leg]
        return tau

    def leg_tau(self, d, leg, force_world):
        """tau (3,) for one leg from a desired world force at the foot: tau = J^T f."""
        return self.foot_jac(d, leg).T @ force_world

    def bias_tau(self, d):
        """Gravity feed-forward G(q) for the 12 actuated joints, in actuator order.
        Essential here: the legs are ~70% of the mass, so without this the legs sag
        and the MPC ground forces are never realised. Coriolis is deliberately
        excluded (its velocity feedback destabilises the dynamic gait)."""
        ds = self._scratch
        ds.qpos[:] = d.qpos
        ds.qvel[:] = 0.0
        mujoco.mj_forward(self.m, ds)                 # qfrc_bias = G(q) when qvel=0
        tau = np.zeros(12)
        tau[self.ctrl.flatten()] = ds.qfrc_bias[self.vadr.flatten()]
        return tau

    # ---------------------------------------------------------------- setup helpers
    def _composite_inertia(self, d):
        """Whole-robot inertia about the CoM, in the LEVEL body frame.

        At the standing keyframe the level frame coincides with the world frame
        (R_level = I), so the world-frame composite inertia here *is* the body-frame
        inertia used by the SRB-MPC. Taken about the SRB point.
        """
        com = self.com(d)
        I = np.zeros((3, 3))
        for b in range(1, self.m.nbody):
            mb = self.m.body_mass[b]
            if mb <= 0:
                continue
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(self.m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - com
            I += Ib + mb * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def _hip_offsets(self, d):
        """Abduction-anchor offsets from the SRB point, in the LEVEL frame (== world at keyframe)."""
        com = self.com(d)
        off = np.zeros((4, 3))
        for li, leg in enumerate(LEG_NAMES):
            num = LEG_JOINT_NUM[leg][0]
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{num}")
            off[li] = d.xanchor[jid] - com
        return off


# ---------------------------------------------------------------- math utils
def mat2rpy(Rm):
    """ZYX intrinsic (yaw-pitch-roll) -> returns (roll, pitch, yaw)."""
    sy = -Rm[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.9999:
        roll = np.arctan2(Rm[2, 1], Rm[2, 2])
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        roll = np.arctan2(-Rm[1, 2], Rm[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
