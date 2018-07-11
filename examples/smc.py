import gym_quadrotor.dynamics as qdyn
from utilities import NumericalDerivative


class SMCControl(object):
    def __init__(self, copter_params):
        self._params = copter_params  # type: qdyn.CopterParams
        self._dtarget = NumericalDerivative()
        self._derror = NumericalDerivative()
        self._ddtarget = NumericalDerivative()

        # config
        self._sign = None
        self._lambda = 1.0
        self._k = 1.0

    def __call__(self, state, target):
        """

        :param qdyn.DynamicsState state:
        :param target:
        :return:
        """
        dt = self._dtarget(target)
        ddt = self._ddtarget(dt)

        error = target - state
        derror = self._derror(error)

        I = self._params.frame_inertia

        common = ddt + self._lambda * error + self._k * self._sign(derror + self._lambda * error)

        # in our approximation dot(theta) = angvel[1]
        R = common[0] - state.net_rotor_speed * self._params.rotor_inertia * state.angular_velocity[1]
        R -= state.angular_velocity[2] * state.angular_velocity[1] * (I[1] - I[2])
        R += state.angular_velocity[0] * self._params.rotational_drag[0]
        R *= I[0] / self._params.arm_length

        P = common[1] + state.net_rotor_speed * self._params.rotor_inertia * state.angular_velocity[0]
        P -= state.angular_velocity[2] * state.angular_velocity[0] * (I[0] - I[2])
        P += state.angular_velocity[1] * self._params.rotational_drag[1]
        P *= I[1] / self._params.arm_length

        Y = common[2] - state.angular_velocity[0] * state.angular_velocity[1] * (I[0] - I[1])
        Y += state.angular_velocity[2] * self._params.rotational_drag[2]
        Y *= I[2] * self._params.thrust_factor / self._params.drag_factor

