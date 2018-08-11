import numpy as np
from gym import spaces
from gym_quadrotor.envs.base import QuadRotorEnvBase
from gym_quadrotor.dynamics.coordinates import angvel_to_euler, angle_difference


# TODO normalize euler angles
# TODO fix observation space
class CopterStabilizeAttitudeEnv(QuadRotorEnvBase):
    observation_space = spaces.Box(0, 1, (6,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self._target_yaw = 0
        self._error_target = 1 * np.pi / 180
        self._correct_counter = 0

    def _step_copter(self, action: np.ndarray):
        attitude = self._state.attitude
        # TODO add another error term penalizing velocities.
        angle_error = attitude.roll**2 + attitude.pitch**2 + angle_difference(attitude.yaw, self._target_yaw)**2
        velocity_error = np.sum(self._state.angular_velocity**2)
        reward = -angle_error
        self.limit_attitude(np.pi/4)
        self.ensure_fixed_position()
        # check whether error is below bound for a given time interval
        if angle_error < self._error_target * self._error_target:
            self._correct_counter += 1
            # after 1 second, win the episode
            if self._correct_counter > 50:
                return 1.0, True, {}
        else:
            self._correct_counter = 0
        return reward, False, {}

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [s.attitude.roll, s.attitude.pitch, angle_difference(s.attitude.yaw, self._target_yaw), rate[0], rate[1], rate[2]]
        return np.array(state)

    def _reset_copter(self):
        self.randomize_angle(20)

        self._target_yaw = self._state.attitude.yaw + self.random_state.uniform(low=-0.3, high=0.3)
        self._state.position[2] = 1
        self._correct_counter = 0

