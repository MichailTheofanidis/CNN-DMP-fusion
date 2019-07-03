#!/usr/bin/env python

import rospy
import roslib
import tf2_ros
import tf
import math
import numpy as np
import random as rn
from world_control_msgs.srv import *
from geometry_msgs.msg import *
roslib.load_manifest('service_pkg')

X_MAX = 0.7
X_MIN = 0.45

Y_MAX = 0.3
Y_MIN = -0.3

YAW_MIN = -1.5708
YAW_MAX = 1.5708

TABLE_SCALE = 0.7
TABLE_HEIGHT = 0.86 * TABLE_SCALE

CONVERT = 100

SPAWN = 4
PAD = 0.045


class Environment:

    def __init__(self):

        self.key_point_length = 50

        self.kitchen_world = "/iai_kitchen/world"

        self.robot_world = "/odom_combined"

        self.arm_world = "/torso_lift_link"

        self.service_name = "/unreal/move_object"

        self.tf_buffer = tf2_ros.Buffer()

        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.names = [{'name': 'SM_BaerenMarkeAlpenfrischerKakao_2', 'x': 135, 'y': -110, 'z': 93, 'h': 0.22, 'B': ['Top', 'Side']},
                      {'name': 'SM_BrandtVollkornZwieback_5', 'x': 170, 'y': -110, 'z': 89, 'h': 0.14, 'B': ['Top']},
                      {'name': 'SM_Cappuccino_8', 'x': 150, 'y': -110, 'z': 88, 'h': 0.13, 'B': ['Top', 'Side']},
                      {'name': 'SM_CoffeeElBryg_11', 'x': 170, 'y': -95, 'z': 92, 'h': 0.17, 'B': ['Top']},
                      {'name': 'SM_HelaCurryKetchup_14', 'x': 150, 'y': -95, 'z': 93, 'h': 0.21, 'B': ['Top', 'Side']},
                      {'name': 'SM_HohesCOrange_17', 'x': 135, 'y': -95, 'z': 97, 'h': 0.24, 'B': ['Top']},
                      {'name': 'SM_JaMilch_20', 'x': 135, 'y': -80, 'z': 90, 'h': 0.20, 'B': ['Top']},
                      {'name': 'SM_JodSalz_23', 'x': 150, 'y': -80, 'z': 90, 'h': 0.14, 'B': ['Top', 'Side']},
                      {'name': 'SM_KoellnMuesliCranberry_29', 'x': 135, 'y': -65, 'z': 93, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_KoellnMuesliKnusperHonigNuss_32', 'x': 155, 'y': -65, 'z': 92, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_MeerSalz_35', 'x': 170, 'y': -65, 'z': 90, 'h': 0.13, 'B': ['Top', 'Side']},
                      {'name': 'SM_MuellerFruchtButterMilchMultiVitamin_41', 'x': 135, 'y': -50, 'z': 90, 'h': 0.17, 'B': ['Top', 'Side']},
                      {'name': 'SM_NesquikCereal_44', 'x': 150, 'y': -50, 'z': 96, 'h': 0.28, 'B': ['Top']},
                      {'name': 'SM_ReineButterMilch_47', 'x': 170, 'y': -50, 'z': 90, 'h': 0.17, 'B': ['Top', 'Side']},
                      {'name': 'SM_SojaMilch_50', 'x': 135, 'y': -35, 'z': 94, 'h': 0.20, 'B': ['Top']},
                      {'name': 'SM_SpitzenReis_53', 'x': 150, 'y': -35, 'z': 90, 'h': 0.14, 'B': ['Top']},
                      {'name': 'SM_VollMilch_56', 'x': 140, 'y': -65, 'z': 93, 'h': 0.20, 'B': ['Top']},
                      {'name': 'SM_WasaDelicateCrispRosemary_59', 'x': 135, 'y': -20, 'z': 89, 'h': 0.13, 'B': ['Top']}]

    def pick_object(self, name):

        index = 0

        for i in range(0, len(self.names)):

            if self.names[i].get('name') is name:
                index = i

        return index

    def pick_random_object(self):

        i = rn.randint(0, len(self.names)-1)

        return self.names[i].get('name'), i

    def pick_random_behavior(self, i):

        index = rn.randint(0, len(self.names[i].get('B')) - 1)

        b = self.names[i].get('B')

        return b[index]

    def find_transform(self, origin, target):

        t = self.tf_buffer.lookup_transform(origin, target, rospy.Time(0), rospy.Duration(10))

        p = [t.transform.translation.x,
             t.transform.translation.y,
             t.transform.translation.z]

        q = [t.transform.rotation.x,
             t.transform.rotation.y,
             t.transform.rotation.z,
             t.transform.rotation.w]

        T = np.array(tf.transformations.quaternion_matrix([q[0], q[1], q[2], q[3]]))
        T[0:3, -1] = p

        return T

    def move_object_random(self, i):

        # Find the relationship between the torso and the ground frame
        T = self.find_transform("odom_combined", "torso_lift_link")

        # Define the spawning location of the object with respect to the torso of the robot
        x = round(rn.uniform(X_MIN, X_MAX), 3)
        y = round(rn.uniform(Y_MIN, Y_MAX), 3)
        z = -((T[2, -1] - TABLE_HEIGHT)-self.names[i].get('h')/2)

        d = [x, y, z, 1]

        # Find the location of the object with respect to the ground frame
        pw = np.dot(T, d)
        pw = pw[0:3]

        # Define the final transformation
        yaw = rn.uniform(YAW_MIN, YAW_MAX)

        orient = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pos = [pw[0]*CONVERT, -pw[1]*CONVERT, pw[2]*CONVERT+SPAWN]
        self.move_object(self.names[i].get('name'), pos, orient)

        return [x, y, z], math.pi-yaw

    def generate_key_points(self, i, pos, yaw):

        b = self.pick_random_behavior(i)

        if b is 'Top':

            offset = self.names[i].get('h')/2
            low = pos[2]+offset+PAD
            high = pos[2]+offset+2*PAD

            k = np.zeros((self.key_point_length, 3))
            zeta = np.linspace(low, high, self.key_point_length)

            for i in range(0, self.key_point_length):
                k[i] = [pos[0], pos[1], zeta[i]]

            o = tf.transformations.quaternion_from_euler(0, 1.571, yaw)

        if b is 'Side':

            offset = self.names[i].get('h')/2
            low = pos[2]+PAD/2
            high = pos[2]+offset/2

            k = np.zeros((self.key_point_length, 3))
            zeta = np.linspace(low, high, self.key_point_length)

            for i in range(0, self.key_point_length):
                k[i] = [pos[0], pos[1]-offset, zeta[i]]

            o = tf.transformations.quaternion_from_euler(0, 0, 1.571)

        return [k, o]

    def move_object_back(self, i):

        pos = [self.names[i].get('x'), self.names[i].get('y'), self.names[i].get('z')]
        orient = [0, 0, 0, 1]

        self.move_object(self.names[i].get('name'), pos, orient)

    def move_object(self, name, p, o):

        rospy.wait_for_service(self.service_name)

        s = rospy.ServiceProxy(self.service_name, MoveObject)

        pos = Pose()
        pos.position.x = p[0]
        pos.position.y = p[1]
        pos.position.z = p[2]

        pos.orientation.x = o[0]
        pos.orientation.y = o[1]
        pos.orientation.z = o[2]
        pos.orientation.w = o[3]

        s(name, pos)

        rospy.sleep(rospy.Duration(1.5))


