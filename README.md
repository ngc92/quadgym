# Quadgym

This repository provides OpenAI gym environments for the simulation
of quadrotor helicopters. The simulation is restricted to just the 
flight physics of a quadrotor, by simulating a simple dynamics model.
In particular, no environment (obstacles, wind) is considered. 

The purpose of these environments is to test low level control algorithms
for quadrotor drones. Utilities to apply classical control algorithms,
such as a PID controller are provided. 

The following `gym.Env` environments are defined in this package:
* `QuadrotorStabilizeAttitude2D-v0` A simplified quadrotor simulation, in which the movements of the
drone are restricted to a single plane. There is only control input, that controls the angular acceleration, 
and the state space consists of the angle and angular velocity.
* `QuadrotorStabilizeAttitude2D-Markovian-v0` This is a version of `QuadrotorStabilizeAttitude2D-v0` in which the
motor dynamics are instantaneous, to make the dynamics Markovian (instead of a partially observable MDP).
* `QuadrotorStabilizeAttitude-MotorCommands-v0` A simulation of a quadrotor on a test stand, i.e. in a situation in
which the position and (linear) velocity are fixed at zero, and only the angular dynamics are simulated. In this version
of the environment, the control input is given by four real numbers between zero and one that define the strength of the
motor control.
* `QuadrotorStabilizeAttitude-MotorCommands-v0` Same as above, but the control input is given as signals in the [-1, 1]
range to define total thrust (ignored in this setting) and the desired angular accelerations for the roll, pitch, and
yaw angles.

