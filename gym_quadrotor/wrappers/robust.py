import gym
import numpy as np
from numbers import Real
from gym_quadrotor.dynamics import CopterParams
from gym_quadrotor.envs import QuadRotorEnvBase


def _random_between(bounds):
    min_, max_ = bounds
    if isinstance(min_, Real):
        return min_ + np.random.rand() * (max_ - min_)
    else:
        return min_ + np.random.random(size=max_.shape) * (max_ - min_)


def _scale_copter_params(params: CopterParams, factor: float):
    """
    Changes all copter parameters by a constant factor
    :param factor:
    :return:
    """
    return CopterParams(*[x * factor for x in params.as_tuple])


class RobustControlWrapper(gym.Wrapper):
    """
    A wrapper that randomizes the copter parameters for each episode. Copter parameters
    are chosen uniformly between a given lower and upper bound.
    """

    def __init__(self, env: gym.Env, lower: CopterParams, upper: CopterParams):
        super().__init__(env)
        assert isinstance(self.unwrapped, QuadRotorEnvBase)
        self.lower = lower
        self.upper = upper

    @staticmethod
    def from_scale(env: gym.Env, lower: float = 0.75, upper: float = 1.33):
        base_params = env.unwrapped.setup
        return RobustControlWrapper(env, _scale_copter_params(base_params, lower),
                                    _scale_copter_params(base_params, upper))

    def reset(self, **kwargs):
        lower = self.lower.as_tuple()
        upper = self.upper.as_tuple()
        random_setup = CopterParams(*map(_random_between, zip(lower, upper)))

        self.unwrapped.setup = random_setup
