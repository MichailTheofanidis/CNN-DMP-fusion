#!/usr/bin/env python

import roslib
import actionlib
from service_pkg.srv import *
from geometry_msgs.msg import Twist
from control_msgs.msg import *
from trajectory_msgs.msg import *
from trac_ik_python.trac_ik import *
from robot_class.utils import *
from pr2_controllers_msgs.msg import *


class MyRobot:

    def __init__(self):

        # Names of robot joints
        self.names = ['torso_lift_joint', 'r_upper_arm_roll_joint', 'r_shoulder_pan_joint', 'r_shoulder_lift_joint',
                      'r_forearm_roll_joint', 'r_elbow_flex_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint',
                      'l_upper_arm_roll_joint', 'l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_forearm_roll_joint',
                      'l_elbow_flex_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint', 'head_pan_joint', 'head_tilt_joint']

        # Right arm names
        self.right_names = self.names[1:8]

        # Left arm names
        self.left_names = self.names[8:15]

        # Dictionary for joint_names
        self.joint_names = {"torso": self.names[0],
                            "r_q0": self.names[1],
                            "r_q1": self.names[2],
                            "r_q2": self.names[3],
                            "r_q3": self.names[4],
                            "r_q4": self.names[5],
                            "r_q5": self.names[6],
                            "r_q6": self.names[7],
                            "l_q0": self.names[8],
                            "l_q1": self.names[9],
                            "l_q2": self.names[10],
                            "l_q3": self.names[11],
                            "l_q4": self.names[12],
                            "l_q5": self.names[13],
                            "l_q6": self.names[14],
                            "head": self.names[15]}

        # Right arm names
        self.joint_right_names = ["r_q0", "r_q1", "r_q2", "r_q3", "r_q4", "r_q5", "r_q6"]

        # Left arm names
        self.joint_left_names = ["l_q0", "l_q1", "l_q2", "l_q3", "l_q4", "l_q5", "l_q6"]

        # Initialize joint state listener
        roslib.load_manifest('service_pkg')

        # Initialize ros node
        rospy.init_node('robot')

        # Topic to receive the state of the robot
        self.state_service = "return_joint_states"

        # Topic to send robot trajectories
        self.topic = '/whole_body_controller/follow_joint_trajectory'

        # Topic to control the robot gripper
        self.gripper_topic = '/r_gripper_controller/gripper_action'

        self.arm_client = actionlib.SimpleActionClient(self.topic, FollowJointTrajectoryAction)

        self.gripper_client = actionlib.SimpleActionClient(self.gripper_topic, Pr2GripperCommandAction)

    # Method that initializes the IK solver of the class
    def ik_init(self, arm="right", param=False):

        if arm == "right":
            solver = IK("torso_lift_link", "r_gripper_tool_frame")

        if arm == "left":
            solver = IK("torso_lift_link", "l_gripper_tool_frame")

        if param is True:
            print("IK solver uses link chain:")
            print(solver.link_names)

            print("IK solver base frame:")
            print(solver.base_link)

            print("IK solver tip link:")
            print(solver.tip_link)

            print("IK solver for joints:")
            print(solver.joint_names)

            print("IK solver using joint limits:")
            lb, ub = solver.get_joint_limits()
            print("Lower bound: " + str(lb))
            print("Upper bound: " + str(ub))

        return solver

    # Get an IK solution
    def ik_solution(self, solver, p, o, q_start, param=True):

        # Boundaries
        bx = by = bz = 0.001

        # Final orientation restriction
        if param is True:
            # High restriction
            brx = bry = brz = 0.1
        else:
            # No restriction
            brx = bry = brz = 9999.0

        sol = solver.get_ik(q_start, p[0], p[1], p[2], o[0], o[1], o[2], o[3],
                            bx, by, bz, brx, bry, brz)
        if sol is not None:
            sol = np.array([sol[2], sol[0], sol[1], sol[4], sol[3], sol[5], sol[6]])

        return sol

    # Constrain the joint angles of the robot
    def ik_constrain(self, q):

        id = []

        for i in range(0, len(q)):

            if q[i] > math.pi:
                id.append(i)

        q[id] = -(2*math.pi-q[id])

        return q

    # Get the state of the Robot
    def get_state(self):

        rospy.wait_for_service(self.state_service)
        s = rospy.ServiceProxy(self.state_service, ReturnJointStates)
        resp = s(self.names)

        return np.array(list(resp.position)), np.array(list(resp.velocity)), np.array(list(resp.effort))

    # Get the state of specific joint states
    def get_joint_state(self, joints):

        q_init, dq_init, ddq_init = self.get_state()
        target_joints = [self.joint_names.get(x) for x in joints]
        index = [self.names.index(x) for x in target_joints]

        return q_init[index], dq_init[index], ddq_init[index]

    # Move the base of the robot
    def move_base(self, target, speed):

        twist = Twist()
        twist.linear.x = speed

        pub = rospy.Publisher('base_controller/command', Twist, queue_size=1)

        rate = rospy.Rate(1)
        counter = 0
        hold = target
        while counter < hold:
            counter += 1
            pub.publish(twist)
            rate.sleep()

        twist.linear.x = 0
        pub.publish(twist)
        rospy.sleep(rospy.Duration(1))

    # Send Trajectory Command
    def arm_control(self, msg):

        self.arm_client.wait_for_server()
        self.arm_client.send_goal(msg)
        self.arm_client.wait_for_result()
        self.arm_client.cancel_all_goals()
        rospy.sleep(rospy.Duration(1))

    # Send Trajectory Command
    def gripper_control(self, msg):

        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal(msg)
        self.gripper_client.wait_for_result()
        self.gripper_client.cancel_all_goals()
        rospy.sleep(rospy.Duration(1))

    # Reverse a trajectory to go back to the original position
    def reverse_traiectory(self, traj, d_traj, dd_traj):
        return np.flipud(traj), np.flipud(d_traj), np.flipud(dd_traj)

    # Design a trajectory from keypoints
    def generate_trajectory(self, joints, key, orient, time):

        # Flag to see if solution is found
        found = True

        # Get the initial joint position
        solver = self.ik_init()

        q_i, dq_i, ddq_i = self.get_joint_state(joints)

        # Find the target joint position
        q_f = None
        cnt = 0
        while q_f is None:

            position = [key[cnt, 0], key[cnt, 1], key[cnt, 2]]
            q_f = self.ik_solution(solver, position, orient, q_i)
            cnt += 1

            if cnt == key.shape[0]:
                q_f = q_i
                found = False

        q_f = self.ik_constrain(q_f)

        traj, d_traj, dd_traj = self.parabolic_trajectory(q_i, q_f, time)

        return traj, d_traj, dd_traj, time, found

    # Design a trajectory with parabolic blends
    def parabolic_trajectory(self, q_i, q_f, time):

        traj = np.zeros((len(time), len(q_i)))
        for i in range(0, len(q_i)):
            c = coefficient(q_i[i], q_f[i], 0, 0, time[-1])
            traj[:, i] = trajectory(c, time)

        d_traj = np.zeros(traj.shape)
        dd_traj = np.zeros(traj.shape)

        for i in range(0, traj.shape[1]):
            d_traj[:, i] = vel(traj[:, i], time)
            dd_traj[:, i] = vel(d_traj[:, i], time)

        return traj, d_traj, dd_traj

    # Design a simple trajectory based on interpolation
    def simple_trajectory(self, joints, q_target, time):

        q_init, dq_init, ddq_init = self.get_joint_state(joints)

        traj = interp_trajectory(q_init, q_target, time)

        d_traj = np.zeros(traj.shape)
        dd_traj = np.zeros(traj.shape)

        for i in range(0, d_traj.shape[1]):
            d_traj[:, i] = vel(traj[:, i], time)
            dd_traj[:, i] = vel(d_traj[:, i], time)

        return traj, d_traj, dd_traj

    # Generate the target trajectory message for the action library
    def generate_message(self, joints, traj_init, d_traj_init, dd_traj_init, time):

        q, dq, ddq = self.get_state()
        target_joints = [self.joint_names.get(x) for x in joints]
        id = [self.names.index(x) for x in target_joints]
        indices = [i for i in range(0, len(self.names)) if i not in id]

        traj = np.zeros((len(time), len(self.names)))
        d_traj = np.zeros(traj.shape)
        dd_traj = np.zeros(traj.shape)

        traj[:, id] = traj_init
        d_traj[:, id] = d_traj_init
        dd_traj[:, id] = dd_traj_init

        traj[:, indices] = q[indices]
        d_traj[:, indices] = dq[indices]
        dd_traj[:, indices] = ddq[indices]

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.names

        for i in range(0, len(time)):
            point = JointTrajectoryPoint()
            point.positions = traj[i]
            point.velocities = d_traj[i]
            point.accelerations = dd_traj[i]
            point.time_from_start = rospy.Duration(time[i])
            goal.trajectory.points.append(point)

        return goal

    # Open gripper message
    def open_msg(self):

        msg = Pr2GripperCommandGoal()
        msg.command.position = 0.08
        msg.command.max_effort = 0.1

        return msg

    # Close gripper message
    def close_msg(self):

        msg = Pr2GripperCommandGoal()
        msg.command.position = 0.0
        msg.command.max_effort = 50.0

        return msg
