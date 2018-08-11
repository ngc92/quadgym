import pytest
from gym_quadrotor.control.utilities import *


def numerical_derivative_test():
    deriv = NumericalDerivative()

    # first step always gives derivative of zero
    assert deriv(1, time=1) == 0

    # unit time step - derivative = change
    assert deriv(1, time=2) == pytest.approx(0)
    assert deriv(2, time=3) == pytest.approx(1)

    # double time step
    assert deriv(1, time=5) == pytest.approx(-0.5)

    # check reset: next result will be zero again
    deriv.reset()
    assert deriv(-1, time=6) == 0
    assert deriv(2, time=6.5) == pytest.approx(6)


def test_numerical_integral():
    integral = NumericalIntegral()

    # by defualt the integral starts with zero
    assert integral(5, 0) == pytest.approx(0)

    # this is a very simple integration routine: use the new value over the complete interval.
    assert integral(5, 1) == pytest.approx(5)
    assert integral(10, 2) == pytest.approx(15)
    assert integral(5, 3) == pytest.approx(20)

    # check resetting
    integral.reset(initial_value=3)
    assert integral(-5, 3) == pytest.approx(3)
    assert integral(2, 5) == pytest.approx(3+4)
