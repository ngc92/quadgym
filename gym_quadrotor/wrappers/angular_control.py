from gym import ActionWrapper, spaces
import numpy as np
from gym_quadrotor.control.utilities import attitude_to_motor_control


class AngularControlWrapper(ActionWrapper):
    action_space = spaces.Box(-np.ones(4), np.ones(4), dtype=np.float32)

    def __init__(self, env, fixed_total=None):
        super(AngularControlWrapper, self).__init__(env)
        self._fixed_total = fixed_total

    def _action(self, action):
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


# TODO re-enable this code
def estimate_hover_offset(setup):
    from gym_quadrotor.envs.copter import calculate_equilibrium_acceleration

    def calculate(control):
        return calculate_equilibrium_acceleration(setup, control).linear[2]

    controls = np.linspace(0.0, 1.0, 100)
    accels = [calculate(c) for c in controls]
    best = np.argmin(np.abs(accels))
    return controls[best]
