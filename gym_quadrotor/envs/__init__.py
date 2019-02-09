from .base import QuadRotorEnvBase
from .attitude_env import CopterStabilizeAttitudeEnv
from .minimal import CopterStabilizeAttitude2DEnv, CopterStabilizeAttitude2DFullStateEnv


def CopterStabilizeAttitudeEnvAngular():
    from gym_quadrotor.wrappers.angular_control import AngularControlWrapper
    return AngularControlWrapper(CopterStabilizeAttitudeEnv(), fixed_total=2.25)


def make_sparse_reward_env_2d(params=None):
    env = CopterStabilizeAttitude2DEnv(params)
    from gym_quadrotor.wrappers.reward import modify_attitude_reward
    modify_attitude_reward(env, 0.0, 0.0)
    return env
