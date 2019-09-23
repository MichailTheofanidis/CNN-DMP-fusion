#!/usr/bin/env python

from robot_class.robot import MyRobot as Robot
from robot_class.utils import *
from robot_class.recorder import Recorder
from robot_class.environment import *
from robot_class.dmp import DynamicMovementPrimitive as DMP


# Initialize time simulation variables
T_MIN = 2
T_MAX = 4
INTERP = 100

DATASET_ID = 1
RUNS = 1

# Start the Simulation
robot = Robot()

# Initialize the IK solver
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

print(directory)

# DMP class initialization
my_dmp = DMP(20.0, 20, False)

# Traverse all the testing items
items = len(env.names)
for index in range(0, items):

    # Print the items name
    print(env.names[index].get('name'))
    print('====================')

    # Empty the storing array
    recorder.empty_array()

    # Make a directory to store the demo images of the object
    recorder.make_image_directory(index)

    # Start different grasping attempts
    for run in range(0, RUNS):

        # Number of run
        print(run)
        print('--------------------')

        # Move the object in a random location on the table
        position, yaw = env.move_object_random(index)

        # Generate the key points of the trajectory
        [k, o] = env.generate_key_points(index, position, yaw)

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
            recorder.trajectory_id = run

            if index == 0 and run ==0:
                # Start the Recorder
                recorder.start()

                # Start Recording
                recorder.resume()
            else:
                # Start Recording
                recorder.resume()

            # Move the Robot near the object
            goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
            robot.arm_control(goal)

            # Concatenate the DMP weight vector
            array = w.T.reshape((1, np.size(w)))
            tile = np.tile(array, (INTERP, 1))

            # Store the joint angles
            recorder.store_angles(traj, d_traj, tile)

            # Stop Recording
            recorder.pause()

            # Wait to finish recording
            rospy.sleep(rospy.Duration(1))

            # Create the gif
            recorder.make_gif(run)

            # Move the robot to home position
            q = radiant([-90, -70, 0, 0, -70, 0, 0])
            joint_names = robot.joint_right_names
            traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
            goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
            robot.arm_control(goal)

        # Move the object back to its original location
        env.move_object_back(index)

        # Record the joint angles
        recorder.store_demo(index)

    print('====================')

