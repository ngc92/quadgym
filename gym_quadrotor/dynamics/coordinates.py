import numpy as np

# http://www.chrobotics.com/library/understanding-euler-angles


class Euler(object):
    """
    Defines an Euler angle (roll, pitch, yaw). We
    do try to cache as many intermediate results
    as possible here (e.g. the transformation matrices).

    Therefore, do not change the `_euler` attribute
    except for using the provided setters!
    """
    def __init__(self, roll, pitch, yaw):
        self._euler = np.array([roll, pitch, yaw])
        self._cache = {}

    @staticmethod
    def from_numpy_array(array):
        array = np.asarray(array)
        assert array.shape == (3,)
        return Euler(array[0], array[1], array[2])

    @staticmethod
    def zero():
        return Euler(0, 0, 0)

    @property
    def roll(self):
        return self._euler[0]

    @roll.setter
    def roll(self, value):
        self._euler[0] = value
        self._cache = {}

    @property
    def pitch(self):
        return self._euler[1]

    @pitch.setter
    def pitch(self, value):
        self._euler[1] = value
        self._cache = {}

    @property
    def yaw(self):
        return self._euler[2]

    @yaw.setter
    def yaw(self, value):
        self._euler[2] = value
        self._cache = {}

    def rotate(self, amount):
        self._euler += amount
        self._cache = {}

    def rotated(self, amount):
        return Euler(self.roll + amount[0], self.pitch + amount[1], self.yaw + amount[2])

    def add_to_cache(self, key, value):
        self._cache[key] = value

    def get_from_cache(self, key):
        return self._cache.get(key)


def body_to_world_matrix(euler):
    """
    Creates a transformation matrix for directions from a body frame
    to world frame for a body with attitude given by `euler` Euler angles.
    :param euler: The Euler angles of the body frame.
    :return: The transformation matrix.
    """
    return np.transpose(world_to_body_matrix(euler))


def world_to_body_matrix(euler):
    """
    Creates a transformation matrix for directions from world frame
    to body frame for a body with attitude given by `euler` Euler angles.
    :param euler: The Euler angles of the body frame.
    :return: The transformation matrix.
    """

    # check if we have a cached result already available
    matrix = euler.get_from_cache("world_to_body")
    if matrix is not None:
        return matrix

    roll = euler.roll
    pitch = euler.pitch
    yaw = euler.yaw

    Cy = np.cos(yaw)
    Sy = np.sin(yaw)
    Cp = np.cos(pitch)
    Sp = np.sin(pitch)
    Cr = np.cos(roll)
    Sr = np.sin(roll)

    matrix = np.array(
        [[Cy * Cp, Sy * Cp, -Sp],
         [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr],
         [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]])

    euler.add_to_cache("world_to_body", matrix)

    return matrix


def body_to_world(euler, vector):
    """
    Transforms a direction `vector` from body to world coordinates, where the body frame
    is given by the Euler angles `euler.
    :param euler: Euler angles of the body frame.
    :param vector: The direction vector to transform.
    :return: Direction in world frame.
    """
    return np.dot(body_to_world_matrix(euler), vector)


def world_to_body(euler, vector):
    """
    Transforms a direction `vector` from world to body coordinates, where the body frame
    is given by the Euler angles `euler.
    :param euler: Euler angles of the body frame.
    :param vector: The direction vector to transform.
    :return: Direction in body frame.
    """
    return np.dot(world_to_body_matrix(euler), vector)


def body_z(euler):
    """
    Transforms the z axis from body to world coordinates.
    :param euler: Euler angles of the body frame.
    :return: The z axis of the body frame in world coordinates.
    """
    return body_to_world(euler, [0, 0, 1])


def world_z(euler):
    """
    Transforms the z axis from world to body coordinates.
    :param euler: Euler angles of the body frame.
    :return: The z axis of the world frame in body coordinates.
    """
    return world_to_body(euler, [0, 0, 1])


def angular_velocity_to_euler_matrix(euler):
    """
    Transformation matrix from angular velocity (in body frame) to
    rate of change of euler angles.
    :param euler: Current attitude.
    :return: Matrix R such that dot(a) = R w
    """
    matrix = euler.get_from_cache("angvel_euler")
    if matrix is not None:
        return matrix

    roll = euler.roll
    pitch = euler.pitch

    Cp = np.cos(pitch)
    Sp = np.sin(pitch)
    Cr = np.cos(roll)
    Sr = np.sin(roll)

    matrix = np.array([
        [1,   0,   -Sp],
        [0,  Cr, Cp*Sr],
        [0, -Sr, Cp*Cr]
    ])
    euler.add_to_cache("angvel_euler", matrix)
    return matrix


def angvel_to_euler(euler, angular_velocity):
    """ calculate the rate of change of euler angles for a given angular velocity."""
    return np.dot(angular_velocity_to_euler_matrix(euler), angular_velocity)


def euler_to_angvel(euler, deuler):
    """calculate the angular velocity given a rate of change for the euler angles."""
    return np.dot(np.linalg.inv(angular_velocity_to_euler_matrix(euler)), deuler)


def normalize_angle(angle):
    """
    Normalizes an angle (in radians) to the interval [0, 2pi].
    :param angle: A possibly non-normalized angle in radians.
    :return: The angle in radians, converted into the interval [0, 2pi]
    """
    return np.remainder(angle, 2 * np.pi)


def angle_difference(a, b):
    """
    Calculate the angular difference between two absolute angles a and b. Takes into account
    that the rotation can happen both clockwise and counterclockwise. The returned value is
    negative if the shorter rotation (for mapping b onto a) happens if b is decreased.
    :param a: The target angle.
    :param b: The angle to be rotated.
    :return: The shortest distance `d` such that `a = b + d (mod 2pi)`.
    """
    n = normalize_angle(a - b)
    # at this point all values are in [0, 2pi]
    if n > np.pi:
        return 2*np.pi - n
    else:
        return n
