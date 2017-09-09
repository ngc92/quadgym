from gym import ActionWrapper, spaces
import numpy as np


class AngularControlWrapper(ActionWrapper):
    def __init__(self, env):
        super(AngularControlWrapper, self).__init__(self, env)
        self.action_space = spaces.Box(np.array([0.0, -1.0, -1.0, -1.0]), np.ones(4))

    def _action(self, action):
        # TODO add tests to show that these arguments are ordered correctly
        total = action[0] * 4
        roll = action[1] / 2  # rotation about x axis
        pitch = action[2] / 2  # rotation about y axis
        yaw = action[3] / 2
        return coupled_motor_action(total, roll, pitch, yaw)

    def _reverse_action(self, action):
        raise NotImplementedError()


def coupled_motor_action(total, roll, pitch, yaw):
    a = total / 4 - pitch / 2 + yaw / 4
    b = total / 4 + pitch / 2 + yaw / 4
    c = total / 4 + roll  / 2 - yaw / 4
    d = total / 4 - roll  / 2 - yaw / 4
    return np.array([a, b, c, d])
