from . import wrappers
from gym.envs.registration import register

register(
    id='Quadrotor-v0',
    entry_point='gym_quadrotor.envs:CopterEnv',
    max_episode_steps=1000
)

register(
    id='QuadrotorHover-v0',
    entry_point='gym_quadrotor.envs:HoverCopterEnv',
    max_episode_steps=1000
)


