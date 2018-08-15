import numpy as np
from gym import spaces

from gym_quadrotor.control.utilities import attitude_to_motor_control
from gym_quadrotor.envs.base import QuadRotorEnvBase, clip_attitude, ensure_fixed_position, project_2d
from gym_quadrotor.dynamics.coordinates import angvel_to_euler, angle_difference


# TODO fix observation space
class CopterStabilizeAttitude2DEnv(QuadRotorEnvBase):
    observation_space = spaces.Box(np.array([-np.pi/4, -10]), np.array([np.pi/4, 10]), dtype=np.float32)
    action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self._error_target = 1 * np.pi / 180
        self._correct_counter = 0

    def _step_copter(self, action: np.ndarray):
        ensure_fixed_position(self._state, 1.0)
        project_2d(self._state)
        reward = self._calculate_reward(self._state)
        if clip_attitude(self._state, np.pi/4):
            reward -= 1

        # after 1 second within error bounds, win the episode
        if self._correct_counter > 50:
            return 0.1 / 0.98, True, {}

        return reward, False, {}

    def _calculate_reward(self, state):
        attitude = state.attitude
        angle_error = attitude.pitch ** 2
        # TODO add another error term penalizing velocities.
        velocity_error = np.sum(state.angular_velocity ** 2)
        #print(velocity_error, " ", angle_error)
        #reward = -np.sqrt(angle_error)
        reward = -angle_error
        # check whether error is below bound and count steps
        if angle_error < self._error_target * self._error_target:
            reward += 0.1
            self._correct_counter += 1
        else:
            self._correct_counter = 0
        return reward

    def _process_action(self, action):
        return attitude_to_motor_control(3, 0, action, 0)

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [s.attitude.pitch, np.clip(rate[1], -10, 10)]
        return np.array(state)

    def _reset_copter(self):
        self.randomize_angle(20)

        self._state.position[2] = 1
        self._correct_counter = 0

