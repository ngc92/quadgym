import numpy as np
from gym_quadrotor.envs.base import QuadRotorEnvBase
from gym_quadrotor.dynamics.coordinates import angvel_to_euler


class CopterStabilizeAttitudeEnv(QuadRotorEnvBase):
    def _step(self, action: np.ndarray):
        pass

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [s.attitude.roll, s.attitude.pitch, s.attitude.yaw, rate[0], rate[1], rate[2]]
        return np.array(state)

    def _reset(self):
        self.randomize_angle(5)
