# a few functions for geometric calculations
import numpy as np

class Quaternion(object):
    def __init__(self, w, x, y, z):
        self._data = np.array([w,x,y,z])
        norm = np.linalg.norm(self._data)
        self._data /= norm
        self._matrix = None

    @property
    def w(self):
        return self._data[0]

    @property
    def x(self):
        return self._data[1]

    @property
    def y(self):
        return self._data[2]

    @property
    def z(self):
        return self._data[3]

    @property
    def rotation_matrix(self):
        # we cache the rotation matrix. This means you should NOT 
        # modify _data directly
        if self._matrix is not None:
            return self._matrix

        x = self.x
        y = self.y
        z = self.z
        w = self.w

        matrix = np.zeros((3, 3))
        matrix[0, 0] = 1 - 2*(y*y + z*z)
        matrix[0, 1] = 2*(x*y - w*z)
        matrix[0, 2] = 2*(x*z - w*y)
        matrix[1, 0] = 2*(x*y - w*z)
        matrix[1, 1] = 1 - 2*(x*x + z*z)
        matrix[1, 2] = 2*(y*z + w*x)
        matrix[2, 0] = 2*(x*z + w*y)
        matrix[2, 1] = 2*(y*z + w*x)
        matrix[2, 2] = 1 - 2*(x*x + y*y)
        self._matrix = matrix
        return matrix



def bot(vector, axis):
    return vector - np.dot(vector, axis) * axis

def make_quaternion(roll, pitch, yaw):
    from math import sin, cos
    croll = cos(roll / 2.0);
    sroll = sin(roll / 2.0);
    cpitch = cos(pitch / 2.0);
    spitch = sin(pitch / 2.0);
    cyaw = cos(yaw / 2.0);
    syaw = sin(yaw / 2.0);
    
    w = croll * cpitch * cyaw + sroll * spitch * syaw;
    x = sroll * cpitch * cyaw - croll * spitch * syaw;
    y = croll * spitch * cyaw + sroll * cpitch * syaw;
    z = croll * cpitch * syaw - sroll * spitch * cyaw;
    return Quaternion(w, x, y, z)
