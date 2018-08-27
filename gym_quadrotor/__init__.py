from . import wrappers
from . import dynamics
from . import envs

from gym.envs.registration import register

#register(
#    id='Quadrotor-v0',
#    entry_point='gym_quadrotor.envs:CopterEnv',
#    max_episode_steps=1000
#)
#
#register(
#    id='QuadrotorHover-v0',
#    entry_point='gym_quadrotor.envs:HoverCopterEnv',
#    max_episode_steps=1000
#)

register(
    id="QuadrotorStabilizeAttitude-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitudeEnv',
    max_episode_steps=500
)

register(
    id="QuadrotorStabilizeAttitude2D-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitude2DEnv',
    max_episode_steps=500
)
