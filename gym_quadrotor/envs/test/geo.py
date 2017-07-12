import gym_quadrotor.envs.geo as geo
import numpy as np

def test_bot():
	for i in range(3):
		axis = np.zeros(3)
		vector = np.array([5, 3, -3])
		result = vector.copy()
		axis[i]   = 1
		result[i] = 0

		assert (geo.bot(vector, axis) == result).all

def test_quaterion_basics():
    # check coordinates
    assert geo.Quaternion(1, 0, 0, 0).w == 1
    assert geo.Quaternion(0, 1, 0, 0).x == 1
    assert geo.Quaternion(0, 0, 1, 0).y == 1
    assert geo.Quaternion(0, 0, 0, 1).z == 1

    q = geo.Quaternion(1, 1, 1, 1)
    assert q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z == 1.0, q

def test_matrix():
    print(geo.make_quaternion(0, np.pi/2,0).rotation_matrix)
    print(np.linalg.det(geo.make_quaternion(0, np.pi/2,0).rotation_matrix))
    for i in range(100):
        a, b, c = np.random.random(3) * np.pi * 2
        q  = geo.make_quaternion(a, b, c)
        rmat = q.rotation_matrix
        print(np.linalg.det(rmat))
        print(q.x, q.y, q.z, q.w)
        print(np.linalg.norm(q._data))
        #assert np.linalg.det(rmat) == 1.0, np.linalg.det(rmat) 
        vec = np.random.random(3)
        tf = rmat.dot(vec)
        print(np.linalg.norm(tf, 2))
        print(np.linalg.norm(vec, 2))
        assert np.linalg.norm(tf, 2) == np.linalg.norm(vec, 2), rmat


# TODO check quaternion and rotation matrix correctness

test_bot()
test_quaterion_basics()
test_matrix()