from gym import ActionWrapper, spaces
import numpy as np
from gym_quadrotor.control.utilities import attitude_to_motor_control


class AngularControlWrapper(ActionWrapper):
    def __init__(self, env, fixed_total=None):
        super(AngularControlWrapper, self).__init__(env)
        self._fixed_total = fixed_total

        if fixed_total is not None:
            self.action_space = spaces.Box(-np.ones(3), np.ones(3), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-np.ones(4), np.ones(4), dtype=np.float32)

    def action(self, action):
        # TODO add tests to show that these arguments are ordered correctly
        if self._fixed_total:
            total = self._fixed_total
            roll = action[0]  # rotation about x axis
            pitch = action[1]  # rotation about y axis
            yaw = action[2]
        else:
            total = action[0]
            roll = action[1]  # rotation about x axis
            pitch = action[2]  # rotation about y axis
            yaw = action[3]
        return attitude_to_motor_control(total, roll, pitch, yaw)

    def _reverse_action(self, action):
        raise NotImplementedError()
