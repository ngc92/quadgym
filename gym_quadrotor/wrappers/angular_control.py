from gym import ActionWrapper, spaces
import numpy as np


class AngularControlWrapper(ActionWrapper):
    def __init__(self, env):
        super(AngularControlWrapper, self).__init__(env)
        self.action_space = spaces.Box(-np.ones(4), np.ones(4))

        # calibration
        self._hover_offset = estimate_hover_offset(self.unwrapped.setup)

    def _action(self, action):
        # TODO add tests to show that these arguments are ordered correctly
        total = action[0] * 4
        roll = action[1] * 2  # rotation about x axis
        pitch = action[2] * 2  # rotation about y axis
        yaw = action[3] * 2
        # total = 0 should be action hover_offset, so rescale
        # total = 1 should be action [1, 1, 1, 1], so rescale
        return coupled_motor_action(total, roll, pitch, yaw) * (1.0 - self._hover_offset) + self._hover_offset

    def _reverse_action(self, action):
        raise NotImplementedError()


def estimate_hover_offset(setup):
    from gym_quadrotor.envs.copter import calculate_equilibrium_acceleration

    controls = np.linspace(0.0, 1.0, 100)
    accels = [calculate_equilibrium_acceleration(setup, c) for c in controls]
    best = np.argmin(np.abs(accels))
    return controls[best]


def coupled_motor_action(total, roll, pitch, yaw):
    a = total / 4 - pitch / 2 + yaw / 4
    b = total / 4 + pitch / 2 + yaw / 4
    c = total / 4 + roll  / 2 - yaw / 4
    d = total / 4 - roll  / 2 - yaw / 4
    return np.array([a, b, c, d])
