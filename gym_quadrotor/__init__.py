from . import wrappers
from . import dynamics
from . import envs

from gym.envs.registration import register


def register_with_durations(id: str, **kwargs):
    name = id[:-3]
    version = id[-2:]
    register(name + "-" + version, max_episode_steps=500, **kwargs)
    register(name + "-Short-" + version, max_episode_steps=250, **kwargs)
    register(name + "-Long-" + version, max_episode_steps=1000, **kwargs)


register_with_durations(
    id="QuadrotorStabilizeAttitude-MotorCommands-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitudeEnv',
)

register_with_durations(
    id="QuadrotorStabilizeAttitude-Angular-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitudeEnvAngular',
)

register_with_durations(
    id="QuadrotorStabilizeAttitude2D-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitude2DEnv',
)

register_with_durations(
    id="QuadrotorStabilizeAttitude2D-Sparse-v0",
    entry_point='gym_quadrotor.envs:make_sparse_reward_env_2d',
)

register_with_durations(
    id="QuadrotorStabilizeAttitude2D-FullState-v0",
    entry_point='gym_quadrotor.envs:CopterStabilizeAttitude2DFullStateEnv',
)
