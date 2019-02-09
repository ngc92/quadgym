from .base import QuadRotorEnvBase
from .attitude_env import CopterStabilizeAttitudeEnv
from .minimal import CopterStabilizeAttitude2DEnv, CopterStabilizeAttitude2DSparseRewardEnv


def CopterStabilizeAttitudeEnvAngular():
    from gym_quadrotor.wrappers.angular_control import AngularControlWrapper
    return AngularControlWrapper(CopterStabilizeAttitudeEnv(), fixed_total=2.25)
