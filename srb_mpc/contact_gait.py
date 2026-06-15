"""Contact-aware gait layer + placement-velocity filter (sim2real-ready).

The clock-driven schedule in gait.py stays UNCHANGED; this module adds the layer
that reconciles it with measured reality. Without it, a purely clock-driven trot
is unstable over time: as the body oscillates, swing feet touch down EARLY, the
swing controller keeps force-tracking its arc into the floor and the MPC treats
the foot as force-free — each early contact injects an unmodeled impulse at gait
frequency, pumping the rock until the robot skates away leaning (~31 deg within
12 s in-place). Promoting late-swing measured contacts to stance breaks that
feedback loop; filtering the Raibert placement velocity removes the residual
gait-frequency limit cycle (0.3 deg / 2 mm drift over 30 s in sim).

sim2real notes:
  - `MujocoFootContact` is the SIM implementation of the contact estimator. On
    hardware, replace it with a class exposing the same `measure() -> bool(4)`
    contract (leg order [FR, FL, RR, RL]) from foot force sensors, sole switches,
    or joint-current spikes. Everything else transfers unchanged.
  - `LowPassVelocity` is parametrised by a TIME CONSTANT (seconds), not a
    per-step blend factor, so it behaves identically at a different control rate.
  - `ContactAwareGait` is pure logic (no MuJoCo) and runs as-is on hardware.
"""
import numpy as np
import mujoco

from .robot import LEG_NAMES


class MujocoFootContact:
    """Measured foot-floor contact from the MuJoCo contact buffer (sim only)."""

    def __init__(self, model):
        self.m = model
        self.foot_geom = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_col")
            for leg in LEG_NAMES])
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def measure(self, d):
        """bool(4), leg order [FR, FL, RR, RL]: foot sphere touching the floor."""
        c = np.zeros(4, bool)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if g1 == self.floor_geom or g2 == self.floor_geom:
                other = g2 if g1 == self.floor_geom else g1
                hit = np.where(self.foot_geom == other)[0]
                if hit.size:
                    c[hit[0]] = True
        return c


class ContactAwareGait:
    """Early-touchdown promotion around an UNCHANGED clock-driven gait.

    A leg whose schedule says swing but whose foot measures ground contact in
    LATE swing (phase > `promote_after`) is promoted to stance immediately and
    latched until the schedule itself turns stance. The MPC and the swing
    controller then both treat it as a support leg, so the early contact carries
    load instead of injecting an impulse.
    """

    def __init__(self, gait, promote_after=0.5):
        self.gait = gait                      # the wrapped clock-driven schedule
        self.promote_after = promote_after
        self.early = np.zeros(4, bool)        # latched early-touchdown legs
        self._lifted = np.zeros(4, bool)      # leg seen airborne this swing

    # ------------------------------------------------------------ per step
    def update(self, t, measured):
        """Reconcile schedule with measured contact.

        A contact only counts as an (early) TOUCHDOWN if the foot has actually
        been airborne during this swing — otherwise a slow liftoff (e.g. jump
        thrust) gets latched straight back to stance and the flight never
        happens. Returns (contact_now bool(4), changed bool); `changed` flags a
        latch transition — the caller should replan the MPC immediately.
        """
        sched = self.gait.contact(t)
        sp = self.gait.swing_phase(t)
        prev = self.early.copy()
        self._lifted = (~sched) & (self._lifted | ~measured)
        self.early = (~sched) & (self.early | ((sp > self.promote_after)
                                               & measured & self._lifted))
        return sched | self.early, bool(np.any(self.early != prev))

    def contact_schedule(self, t, N, dt):
        """Clock schedule with the latched early feet folded into slice 0."""
        sched = self.gait.contact_schedule(t, N, dt)
        sched[0] |= self.early
        return sched

    def reset(self):
        self.early[:] = False
        self._lifted[:] = False

    # ------------------------------------------------- passthroughs (one handle)
    @property
    def stand(self):
        return self.gait.stand

    def swing_phase(self, t):
        return self.gait.swing_phase(t)

    @property
    def t_stance(self):
        return self.gait.t_stance


class LowPassVelocity:
    """First-order low-pass for the Raibert placement velocity.

    The instantaneous CoM velocity oscillates at gait frequency during stepping;
    feeding it raw into the touchdown target wobbles the footholds in resonance
    and sustains a rocking limit cycle. Time-constant parametrisation keeps the
    behaviour identical under a different control rate (sim2real).
    """

    def __init__(self, time_constant=0.10):
        self.tau = time_constant
        self.v = np.zeros(3)

    def update(self, v, dt):
        alpha = dt / (self.tau + dt)
        self.v += alpha * (np.asarray(v) - self.v)
        return self.v.copy()

    def reset(self, v0=None):
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, float).copy()
