import gym
from gym_quadrotor.envs import QuadRotorEnvBase
from gym_quadrotor.envs.reward import AttitudeReward, _NoArgumentPassed


class DiscourageLongEpisodesWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)


def modify_attitude_reward(env: gym.Env, angle_factor=None, angvel_factor=None,
                           angle_error_transform=_NoArgumentPassed,
                           angvel_error_transform=_NoArgumentPassed):
        reward = env.unwrapped._attitude_reward  # type: AttitudeReward
        reward.update_parameters(
            angle_factor=angle_factor,
            angvel_factor=angvel_factor,
            angle_error_transform=angle_error_transform,
            angvel_error_transform=angvel_error_transform
        )

