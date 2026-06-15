"""Trot gait scheduler.

Leg order [FR, FL, RR, RL]. Trot moves diagonal pairs together:
FR+RL (offset 0) and FL+RR (offset 0.5). A leg is in stance while its local phase
is below `duty`, otherwise swinging. The same schedule drives the MPC contact
constraints and the swing-leg controller.
"""
import numpy as np


class TrotGait:
    def __init__(self, period=0.42, duty=0.6):   # DOG4.6 platform retune (was 0.34 for
        #  DOG3.0): the heavier 7.33 kg / 7 N*m robot rocks over in a fast trot; slowing the
        #  gait (longer stance) is the single shared knob that rescues the SLQ+leg stack
        #  (X2 staircase 0.1->0.5). Applied identically to all four stacks via this default.
        self.period = period
        self.duty = duty
        self.offset = np.array([0.0, 0.5, 0.5, 0.0])   # FR, FL, RR, RL
        self.stand = False                              # if True, all feet planted

    @property
    def t_stance(self):
        return self.period * self.duty

    @property
    def t_swing(self):
        return self.period * (1.0 - self.duty)

    def leg_phase(self, t):
        return (t / self.period + self.offset) % 1.0    # (4,) in [0,1)

    def contact(self, t):
        if self.stand:
            return np.ones(4, bool)
        return self.leg_phase(t) < self.duty            # (4,) bool

    def contact_schedule(self, t, N, dt):
        """(N,4) bool contact grid over the MPC horizon."""
        if self.stand:
            return np.ones((N, 4), bool)
        return np.array([self.contact(t + k * dt) for k in range(N)])

    def swing_phase(self, t):
        """Per-leg swing progress in [0,1); 0 for stance legs."""
        lp = self.leg_phase(t)
        sp = (lp - self.duty) / (1.0 - self.duty)
        return np.where(lp >= self.duty, np.clip(sp, 0.0, 1.0), 0.0)
