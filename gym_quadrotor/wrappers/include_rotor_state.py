from gym import ObservationWrapper, spaces, Env
import numpy as np
from gym_quadrotor.control.utilities import attitude_to_motor_control
from gym_quadrotor.envs import QuadRotorEnvBase


class IncludeRotorStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, QuadRotorEnvBase)

    def observation(self, observation):
        env = self.unwrapped   # type: QuadRotorEnvBase
        rotor_speeds = env.get_copter_state().rotor_speeds / env.setup.max_rotor_speed
        return np.concatenate((observation, rotor_speeds))
