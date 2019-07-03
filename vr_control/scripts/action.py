#!/usr/bin/env python

from robot_class.robot import MyRobot as Robot
from robot_class.utils import *
from robot_class.recorder import Recorder
from robot_class.environment import *
from robot_class.dmp import DynamicMovementPrimitive as DMP

# Initialize simulation variables

T_MIN = 2
T_MAX = 4
INTERP = 30

DATASET_ID = 2

RUNS = 50

# Start the Simulation
robot = Robot()

solver = robot.ik_init()

# Initialize the time vector
t = time_vector(T_MAX, INTERP)

# Move the torso of the body up
q = [0.27]
joint_names = ["torso"]
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Robot Torso is up")

# Move the left hand of the robot out of the way
q = [0, 90, 0, 0, 0, 0, 0]
q = radiant(q)
joint_names = robot.joint_left_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)

q = [0, 90, 80, 0, -130, -70, 0]
q = radiant(q)
joint_names = robot.joint_left_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Left Hand is out of the way")

# Move the right hand of the robot in the home position
q = [-90, -70, 0, 0, -70, 0, 0]
q = radiant(q)
joint_names = robot.joint_right_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Right Hand is in home position")

# Open the robot right gripper
grp_msg = robot.open_msg()
robot.gripper_control(grp_msg)
print("Gripper is open")

# Initialize the environment
env = Environment()

# Initialize the Recorder
recorder = Recorder(DATASET_ID)

# Make directory to store data
directory = recorder.make_directory()

# Start the Recorder
recorder.start()

# DMP class initialization
my_dmp = DMP(20.0, 20, False)

for i in range(0, RUNS):

    # Print the number of runs
    print(i)

    # Pick a random object
    name, index = env.pick_random_object()
    print(name)

    # Move the object in a random location on the table
    position, yaw = env.move_object_random(index)

    # Generate the key points of the trajectory
    [k, o] = env.generate_key_points(index, position, yaw)

    # New random time vector
    t = time_vector(rn.uniform(T_MIN, T_MAX), INTERP)

    # Move the Robot near the object
    joint_names = robot.joint_right_names
    traj, d_traj, dd_traj, t, found = robot.generate_trajectory(joint_names, k, o, t)

    if found is True:

        # Get the phase from the time vector
        s = my_dmp.phase(t)

        # Get the Gaussian
        psv = my_dmp.distributions(s)

        # Learn the DMP parameters
        ftarget = np.zeros(traj.shape)
        w = np.zeros((my_dmp.ng, traj.shape[1]))

        for j in range(traj.shape[1]):
            ftarget[:, j], w[:, j] = my_dmp.imitate(traj[:, j], d_traj[:, j], dd_traj[:, j], t, s, psv)

        # Set Recorder parameters
        recorder.trajectory_id = i

        # Start Recording
        recorder.resume()

        # Move the Robot near the object
        goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
        robot.arm_control(goal)

        # Stop Recording
        recorder.pause()

        # Move the robot to home position
        q = [-90, -70, 0, 0, -70, 0, 0]
        q = radiant(q)
        joint_names = robot.joint_right_names
        traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
        goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
        robot.arm_control(goal)

        # Record the dmp parameters and the trajectory
        np.savetxt(directory+'/traj_' + str(i) + '.txt', traj, delimiter=',')
        np.savetxt(directory+'/dmp_' + str(i) + '.txt', w, delimiter=',')

    # Move the object back to its original location
    env.move_object_back(index)

