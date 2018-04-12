import numpy as np
from gym import spaces
from gym_quadrotor.envs.base import QuadRotorEnvBase
from gym_quadrotor.dynamics.coordinates import angvel_to_euler


# TODO normalize euler angles
# TODO fix observation space
class CopterStabilizeAttitudeEnv(QuadRotorEnvBase):
    observation_space = spaces.Box(0, 1, (6,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self._target_yaw = 0

    def _step(self, action: np.ndarray):
        attitude = self._state.attitude
        reward = -attitude.roll**2 - attitude.pitch**2 - (attitude.yaw - self._target_yaw)**2
        return reward, False, {}

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [s.attitude.roll, s.attitude.pitch, s.attitude.yaw, rate[0], rate[1], rate[2]]
        return np.array(state)

    def _reset(self):
        self.randomize_angle(5)
        self._target_yaw = self._state.attitude.yaw
