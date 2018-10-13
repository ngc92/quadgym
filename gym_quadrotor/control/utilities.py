import numpy as np


class NumericalDerivative(object):
    def __init__(self):
        self._old_value = None
        self._old_time  = None
        self._last_result = None

    def __call__(self, value, time):
        if self._old_value is None:
            self._old_value = value
            self._old_time = time
            return 0

        delta_v = value - self._old_value
        delta_t = time - self._old_time
        self._old_time = time
        self._old_value = value
        self._last_result = delta_v / delta_t
        return self._last_result

    def reset(self):
        self._old_value = None
        self._old_time = None
        self._last_result = None


class NumericalIntegral(object):
    def __init__(self, initial_value=0.0):
        self._value = initial_value
        self._old_time = None

    def __call__(self, value, time):
        if self._old_time is None:
            self._old_time = time
            return self._value

        dt = time - self._old_time
        self._old_time = time
        self._value += value * dt
        return self._value

    def reset(self, initial_value=0.0):
        self._value = initial_value
        self._old_time = None


def attitude_to_motor_control(thrust_ctrl: float, roll_ctrl: float, pitch_ctrl: float, yaw_ctrl: float):
    return np.array([
        thrust_ctrl + 2 * pitch_ctrl - yaw_ctrl,
        thrust_ctrl - 2 * roll_ctrl + yaw_ctrl,
        thrust_ctrl - 2 * pitch_ctrl - yaw_ctrl,
        thrust_ctrl + 2 * roll_ctrl + yaw_ctrl
    ], dtype=np.float32) / 4.0
