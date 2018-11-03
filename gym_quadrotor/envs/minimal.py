from typing import Optional

import numpy as np
from gym import spaces

from gym_quadrotor.control.utilities import attitude_to_motor_control
from gym_quadrotor.dynamics import CopterParams
from gym_quadrotor.envs.base import QuadRotorEnvBase, clip_attitude, ensure_fixed_position, project_2d
from gym_quadrotor.dynamics.coordinates import angvel_to_euler, angle_difference
from gym_quadrotor.envs.reward import AttitudeReward


MAX_AVEL = 9


# TODO fix observation space
class CopterStabilizeAttitude2DEnv(QuadRotorEnvBase):
    observation_space = spaces.Box(np.array([-np.pi/4, -MAX_AVEL]), np.array([np.pi/4, MAX_AVEL]), dtype=np.float32)
    action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)

    def __init__(self, params: Optional[CopterParams]=None):
        super().__init__(params)
        self._error_target = 1 * np.pi / 180
        self._in_target_reward = 0.1
        self._boundary_penalty = 1.0
        self._attitude_reward = AttitudeReward(1.0, 1e-2)

    def _step_copter(self, action: np.ndarray):
        ensure_fixed_position(self._state, 1.0)
        project_2d(self._state)

        attitude = self._state.attitude
        angle_error = attitude.pitch ** 2

        reward = self._calculate_reward(angle_error)

        if clip_attitude(self._state, np.pi/4):
            reward -= self._boundary_penalty

        return reward, False, {}

    def _calculate_reward(self, angle_error):
        reward = self._attitude_reward.calculate_reward(self._state)

        # check whether error is below bound and add bonus reward
        if angle_error < self._error_target * self._error_target:
            reward += self._in_target_reward
        return reward

    def _process_action(self, action):
        return attitude_to_motor_control(2.25, 0, action, 0)

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [s.attitude.pitch, np.clip(rate[1], -MAX_AVEL, MAX_AVEL)]
        return np.array(state)

    def _reset_copter(self):
        self.randomize_angle(20)
        self.randomize_angular_velocity(2.0)
        project_2d(self._state)

        self._state.position[2] = 1
        self._correct_counter = 0


class CopterStabilizeAttitude2DMarkovianEnv(CopterStabilizeAttitude2DEnv):
    def __init__(self):
        super().__init__()
        self.setup._rotor_speed_half_time = 1e-5
