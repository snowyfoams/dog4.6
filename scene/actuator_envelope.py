"""MG5010-i10 torque-speed envelope for a CURRENT-CONTROLLED (FOC, torque-mode) servo.

Physics: the current loop raises phase voltage to cancel back-EMF, so below voltage
saturation the actuator delivers commanded torque with ~zero EMF damping. The limits:
  - current limit:  |tau| <= 7 N*m            (joint side; motor 0.7 N*m x 10:1)
  - power limit:    tau * omega <= 160 W       (driving quadrant)
  - voltage cliff:  tau <= B_CLIFF*(W0 - omega), slope B_CLIFF = N^2*kt^2/R_phase
                    = 100 * 0.1^2 / 0.466 = 2.146 N*m*s/rad, zero at W0 = 320 rpm
Driving-quadrant limit: hi(w) = min(TAU, P_MAX/w [w>0], max(0, B_CLIFF*(W0-w)))
Braking quadrant is regenerative -> only the current limit binds: lo(w) = -hi(-w).
Corners: flat to 22.9 rad/s, power-limited to 31.1, cliff to 33.51 rad/s.

Implemented as a MuJoCo user bias callback on <general gaintype="fixed" gainprm="1"
biastype="user">: force = ctrl + bias = clip(ctrl, lo, hi). If install() is never
called, bias = 0 and the actuator degrades to the ideal +/-7 torque source (DOG4.0).
"""
import mujoco

TAU = 7.0
P_MAX = 160.0
KT = 0.1            # N*m/A motor side
R_PHASE = 0.466     # ohm
N_GEAR = 10.0
B_CLIFF = N_GEAR ** 2 * KT ** 2 / R_PHASE        # 2.146 N*m*s/rad (joint side)
W0 = 320.0 * 2 * 3.141592653589793 / 60.0        # 33.51 rad/s (320 rpm, joint side)


def _hi(w):
    """Max driving torque at joint velocity w (positive direction)."""
    lim = TAU
    if w > 0.0:
        lim = min(lim, P_MAX / w, B_CLIFF * (W0 - w))
        if lim < 0.0:
            lim = 0.0
    return lim


def envelope_clip(tau_cmd, w):
    hi = _hi(w)
    lo = -_hi(-w)
    return min(max(tau_cmd, lo), hi)


def _bias_cb(m, d, aid):
    c = d.ctrl[aid]
    return envelope_clip(c, d.actuator_velocity[aid]) - c


def install():
    mujoco.set_mjcb_act_bias(_bias_cb)


def uninstall():
    mujoco.set_mjcb_act_bias(None)
