import gym
import numpy as np
from numbers import Real
from gym_quadrotor.dynamics import CopterParams


def _random_between(bounds):
    min_, max_ = bounds
    if isinstance(min_, Real):
        return min_ + np.random.rand() * (max_ - min_)
    else:
        return min_ + np.random.random(size=max_.shape) * (max_ - min_)


class RobustControlWrapper(gym.Wrapper):
    def __init__(self, env, lower: CopterParams, upper: CopterParams):
        super().__init__(env)
        self.lower = lower
        self.upper = upper

    def reset(self, **kwargs):
        lower = self.lower.as_tuple()
        upper = self.upper.as_tuple()
        random_setup = CopterParams(*map(_random_between, zip(lower, upper)))

        self.unwrapped.setup = random_setup
