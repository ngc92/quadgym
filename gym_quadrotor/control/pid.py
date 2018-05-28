from gym_quadrotor.control.utilities import NumericalDerivative, NumericalIntegral


class PIDControl(object):
    def __init__(self, P, I, D):
        self._P = P
        self._I = I
        self._D = D

        self._deriv = NumericalDerivative()
        self._int = NumericalIntegral()

    def __call__(self, current, target, time):
        error = target - current

        d = self._deriv(current, time)
        i = self._int(error, time)

        return self._P * error - self._D * d + self._I * i
