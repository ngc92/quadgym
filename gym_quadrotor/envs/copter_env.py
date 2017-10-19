import math

from gym import spaces
from gym_quadrotor.envs.copter_env_base import CopterEnvBase
from gym_quadrotor.envs.tasks import StayAliveTask, FlySmoothlyTask, HoldAngleTask, HoverTask

from .copter import *


class CopterEnv(CopterEnvBase):
    #reward_range = (-1.0, 1.0)

    def __init__(self):
        # prepare the tasks
        stayalive = StayAliveTask(weight=1.0)
        smooth    = FlySmoothlyTask(weight=0.2)
        # TODO for now we pass self along to have consistent random
        holdang = HoldAngleTask(5 * math.pi / 180, 25 * math.pi / 180, self, weight=1.0)
        super(CopterEnv, self).__init__(tasks = [holdang, stayalive])

        high = np.array([np.inf]*13)
        
        self.observation_space = spaces.Box(-high, high)
    
    def _on_step(self):
        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.rotor_speeds += self.np_random.uniform(low=-2, high=2, size=(4,))


class HoverCopterEnv(CopterEnvBase):
    # reward_range = (-1.0, 1.0)

    def __init__(self):
        # prepare the tasks
        stay_alive = StayAliveTask(weight=1.0)
        smooth = FlySmoothlyTask(weight=0.05)
        hover = HoverTask(5 * math.pi / 180, 25 * math.pi / 180, weight=1.0)
        super(HoverCopterEnv, self).__init__(tasks=[smooth, stay_alive, hover])

        high = np.array([np.inf] * 10)

        self.observation_space = spaces.Box(-high, high)

    def _on_step(self):
        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.rotor_speeds += self.np_random.uniform(low=-2, high=2, size=(4,))
