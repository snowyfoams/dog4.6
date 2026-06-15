"""Scripted fixed-place jump (pronk): crouch -> thrust -> flight -> land.

`JumpManeuver` duck-types the gait interface that `ContactAwareGait` and
`SwingController` consume (contact, swing_phase, contact_schedule, t_stance,
stand), so a controller can swap it in for `TrotGait` for the duration of one
jump without any change to the fixed trot schedule. It also provides the
time-varying height reference `z_ref(t) -> (z, vz)` consumed by the
controllers' optional `z_ref_fn` hook: a smooth crouch, a thrust ramp to the
ballistic liftoff velocity v_lo = g*T_flight/2, the ballistic arc itself, and a
settle back to standing height.

All four legs share one phase: stance through crouch/thrust/land, swing during
flight (the shared swing arc then carries the feet from their liftoff positions
to touchdown points under the hips — landing preparation for free). The
contact-aware layer's early-touchdown promotion handles the actual landing
instant exactly as it does in the trot.
"""
import numpy as np


class JumpManeuver:
    def __init__(self, t0, h_stand, flight_t=0.30, crouch_t=0.30, thrust_t=0.12,
                 land_t=0.40, crouch_depth=0.05, g=9.81):
        self.t0 = t0
        self.h = h_stand
        self.depth = crouch_depth
        self.g = g
        self.Tc, self.Tt, self.Tf, self.Tl = crouch_t, thrust_t, flight_t, land_t
        self.v_lo = g * flight_t / 2.0            # ballistic liftoff velocity
        # phase boundaries (relative to t0)
        self.t_thrust = self.Tc
        self.t_liftoff = self.Tc + self.Tt
        self.t_land = self.t_liftoff + self.Tf
        self.t_end = self.t_land + self.Tl
        self.stand = False                         # gait-interface flag

    # ------------------------------------------------------- gait interface
    def _tau(self, t):
        return t - self.t0

    def contact(self, t):
        tau = self._tau(t)
        in_flight = self.t_liftoff <= tau < self.t_land
        return np.full(4, not in_flight, bool)

    def swing_phase(self, t):
        tau = self._tau(t)
        if self.t_liftoff <= tau < self.t_land:
            s = (tau - self.t_liftoff) / self.Tf
            return np.full(4, np.clip(s, 0.0, 1.0))
        return np.zeros(4)

    def contact_schedule(self, t, N, dt):
        return np.array([self.contact(t + k * dt) for k in range(N)])

    @property
    def t_stance(self):
        return self.Tl                             # Raibert term during flight

    def done(self, t):
        return self._tau(t) >= self.t_end

    # --------------------------------------------------------- z reference
    def z_ref(self, t):
        """(z, vz) reference at absolute controller time t."""
        tau = self._tau(t)
        if tau < 0.0:
            return self.h, 0.0
        if tau < self.t_thrust:                    # smooth crouch
            s = tau / self.Tc
            a = 3 * s ** 2 - 2 * s ** 3
            da = (6 * s - 6 * s ** 2) / self.Tc
            return self.h - self.depth * a, -self.depth * da
        if tau < self.t_liftoff:                   # thrust: ramp up to v_lo
            s = (tau - self.t_thrust) / self.Tt
            z0 = self.h - self.depth
            # constant-acceleration ramp from 0 to v_lo over Tt
            vz = self.v_lo * s
            z = z0 + 0.5 * self.v_lo * self.Tt * s ** 2
            return z, vz
        if tau < self.t_land:                      # ballistic flight
            ts = tau - self.t_liftoff
            z0 = self.h - self.depth + 0.5 * self.v_lo * self.Tt
            return (z0 + self.v_lo * ts - 0.5 * self.g * ts ** 2,
                    self.v_lo - self.g * ts)
        # landing settle: ease back to standing height
        s = np.clip((tau - self.t_land) / self.Tl, 0.0, 1.0)
        a = 3 * s ** 2 - 2 * s ** 3
        z_td = self.h - self.depth + 0.5 * self.v_lo * self.Tt   # ~touchdown height
        return z_td + (self.h - z_td) * a, 0.0


def install(ctrl, flight_t=0.30, **kw):
    """Swap a JumpManeuver into a running controller (either stack)."""
    jm = JumpManeuver(ctrl.t, ctrl.rob.nominal_h, flight_t=flight_t, **kw)
    ctrl._saved_gait = ctrl.gait
    ctrl._saved_promote = ctrl.cgait.promote_after
    ctrl.gait = jm
    ctrl.cgait.gait = jm
    ctrl.swing.gait = jm
    # in trot, early-swing contact is usually scuffing (gate at 0.5); in a pronk
    # ANY contact after liftoff IS the landing — promote almost immediately, or
    # an early touchdown lands on force-disabled legs and the body grounds out.
    ctrl.cgait.promote_after = 0.15
    ctrl.z_ref_fn = jm.z_ref
    ctrl._since = ctrl.replan_every                # replan with the new schedule now
    return jm


def restore(ctrl, stand=True):
    """Put the trot gait back after the jump (stand mode by default)."""
    g = ctrl._saved_gait
    g.stand = stand
    ctrl.gait = g
    ctrl.cgait.gait = g
    ctrl.swing.gait = g
    ctrl.cgait.promote_after = getattr(ctrl, "_saved_promote", 0.5)
    ctrl.z_ref_fn = None
    ctrl._since = ctrl.replan_every
