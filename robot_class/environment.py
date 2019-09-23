#!/usr/bin/env python

import rospy
import roslib
import tf2_ros
import math
import random
import numpy as np
import tf as transform
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
PAD = 0.05


class Environment:

    def __init__(self):

        self.key_point_length = 50

        self.kitchen_world = "/iai_kitchen/world"

        self.robot_world = "/odom_combined"

        self.arm_world = "/torso_lift_link"

        self.service_name = "/unreal/move_object"

        self.tf_buffer = tf2_ros.Buffer()

        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.names = [{'name': 'SM_BaerenMarkeAlpenfrischerKakao_3', 'x': 50, 'y': 190, 'z': 9, 'h': 0.23, 'B': ['Side']},
                      {'name': 'SM_BaerenMarkeFrischeAlpenmilch18_6', 'x': 30, 'y': 190, 'z': 9, 'h': 0.23, 'B': ['Side']},
                      {'name': 'SM_BrandtVollkornZwieback_9', 'x': 10, 'y': 190, 'z': 5, 'h': 0.15, 'B': ['Top']},
                      {'name': 'SM_Cappuccino_12', 'x': -10, 'y': 190, 'z': 3, 'h': 0.13, 'B': ['Side']},
                      {'name': 'SM_CoffeeElBryg_15', 'x': -30, 'y': 190, 'z': 8, 'h': 0.18, 'B': ['Top']},
                      {'name': 'SM_HelaCurryKetchup_18', 'x': -50, 'y': 190, 'z': 9, 'h': 0.21, 'B': ['Side']},
                      {'name': 'SM_HohesCOrange_21', 'x': -70, 'y': 190, 'z': 12, 'h': 0.24, 'B': ['Top']},
                      {'name': 'SM_JaMilch_24', 'x': -90, 'y': 190, 'z': 6, 'h': 0.20, 'B': ['Top']},
                      {'name': 'SM_JodSalz_27', 'x': -110, 'y': 190, 'z': 5, 'h': 0.15, 'B': ['Top']},
                      {'name': 'SM_KnusperSchokoKeks_30', 'x': -130, 'y': 190, 'z': 10, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_KoellnMuesliCranberry_33', 'x': -150, 'y': 190, 'z': 9, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_KoellnMuesliKnusperHonigNuss_36', 'x': -170, 'y': 190, 'z': 9, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_LionCereal_39', 'x': -190, 'y': 190, 'z': 11, 'h': 0.22, 'B': ['Top']},
                      {'name': 'SM_MeerSalz_42', 'x': -210, 'y': 190, 'z': 7, 'h': 0.16, 'B': ['Side']},
                      {'name': 'SM_MilramButterMilchDrinkErdbeere_45', 'x': -230, 'y': 190, 'z': 9, 'h': 0.20, 'B': ['Side']},
                      {'name': 'SM_MuellerFruchtButterMilchMultiVitamin_54', 'x': -250, 'y': 190, 'z': 7, 'h': 0.17, 'B': ['Side']},
                      {'name': 'SM_MuellerReineButterMilch_57', 'x': -270, 'y': 190, 'z': 7, 'h': 0.17, 'B': ['Side']},
                      {'name': 'SM_MyMuesli_Whole_60', 'x': -290, 'y': 190, 'z': 9, 'h': 0.28, 'B': ['Side']},
                      {'name': 'SM_NesquikCereal_63', 'x': -320, 'y': 190, 'z': 12, 'h': 0.28, 'B': ['Top']},
                      {'name': 'SM_PfannerGruneIcetea_66', 'x': -350, 'y': 190, 'z': 15, 'h': 0.27, 'B': ['Side']},
                      {'name': 'SM_PfannerPfirschIcetea_69', 'x': -370, 'y': 190, 'z': 15, 'h': 0.27, 'B': ['Side']},
                      {'name': 'SM_ReineButterMilch_72', 'x': -390, 'y': 190, 'z': 7, 'h': 0.17, 'B': ['Side']},
                      {'name': 'SM_SojaMilch_84', 'x': -410, 'y': 190, 'z': 11, 'h': 0.20, 'B': ['Top']},
                      {'name': 'SM_SpitzenReis_87', 'x': -430, 'y': 190, 'z': 8, 'h': 0.14, 'B': ['Top']},
                      {'name': 'SM_TomatoAlGustoBasilikum_75', 'x': -450, 'y': 190, 'z': 6, 'h': 0.11, 'B': ['Top']},
                      {'name': 'SM_VollMilch_78', 'x': -470, 'y': 190, 'z': 10, 'h': 0.21, 'B': ['Top']},
                      {'name': 'SM_WasaDelicateCrispRosemary_81', 'x': -490, 'y': 190, 'z': 7, 'h': 0.13, 'B': ['Top']}]

        self.test_names = [{'name': 'SM_AlbiHimbeerJuice_90', 'x': 50, 'y': 250, 'z': 12, 'h': 0.25, 'B': 'Side'},
                           {'name': 'SM_BaerenMarkeAlpenfrischerKakao2_93', 'x': 30, 'y': 250, 'z': 7, 'h': 0.23, 'B': 'Side'},
                           {'name': 'SM_JaNougatBits_96', 'x': 0, 'y': 250, 'z': 11, 'h': 0.26, 'B': 'Top'},
                           {'name': 'SM_MuellerFruchtButterMilchHimbeere_102', 'x': -30, 'y': 250, 'z': 6, 'h': 0.17, 'B': 'Side'}]

        self.test_x = [0.488, 0.682, 0.608, 0.615, 0.514, 0.488, 0.627, 0.52, 0.571, 0.627]

        self.test_y = [0.009, -0.029, 0.07, -0.255, 0.212, 0.147, -0.123, -0.285, 0.038, 0.183]

        self.test_yaw = [-0.0105385499233, 0.0963582519757, -0.232002173728, -1.52230055296, -0.633350652396,
                         -0.82068569816, 0.905739921404, 0.392809829363, 0.801009691457, 0.96618651743]

    def pick_object(self, name):

        index = 0

        for i in range(0, len(self.names)):

            if self.names[i].get('name') is name:
                index = i

        return index

    def pick_random_object(self):

        i = random.randint(0, len(self.names)-1)

        return self.names[i].get('name'), i

    def pick_random_behavior(self, i):

        index = random.randint(0, len(self.names[i].get('B')) - 1)

        b = self.names[i].get('B')

        return b[index]

    def pick_test_behavior(self, i):

        b = self.test_names[i].get('B')

        return b

    def find_transform(self, origin, target):

        t = self.tf_buffer.lookup_transform(origin, target, rospy.Time(0), rospy.Duration(10))

        p = [t.transform.translation.x,
             t.transform.translation.y,
             t.transform.translation.z]

        q = [t.transform.rotation.x,
             t.transform.rotation.y,
             t.transform.rotation.z,
             t.transform.rotation.w]

        T = np.array(transform.transformations.quaternion_matrix([q[0], q[1], q[2], q[3]]))
        T[0:3, -1] = p

        return T

    def find_final_transform(self, origin, target):

        t = self.tf_buffer.lookup_transform(origin, target, rospy.Time(0), rospy.Duration(10))

        p = [t.transform.translation.x,
             t.transform.translation.y,
             t.transform.translation.z]

        q = [t.transform.rotation.x,
             t.transform.rotation.y,
             t.transform.rotation.z,
             t.transform.rotation.w]

        return p, q

    def move_object_random(self, i):

        # Find the relationship between the torso and the ground frame
        T = self.find_transform("odom_combined", "torso_lift_link")

        # Define the spawning location of the object with respect to the torso of the robot
        x = round(random.uniform(X_MIN, X_MAX), 3)
        y = round(random.uniform(Y_MIN, Y_MAX), 3)
        z = -((T[2, -1] - TABLE_HEIGHT)-self.names[i].get('h')/2)

        d = [x, y, z, 1]

        # Find the location of the object with respect to the ground frame
        pw = np.dot(T, d)
        pw = pw[0:3]

        # Define the final transformation
        yaw = random.uniform(YAW_MIN, YAW_MAX)

        orient = transform.transformations.quaternion_from_euler(0, 0, yaw)
        pos = [pw[0]*CONVERT, -pw[1]*CONVERT, pw[2]*CONVERT+SPAWN]
        self.move_object(self.names[i].get('name'), pos, orient)

        return [x, y, z], math.pi-yaw

    def move_test_object(self, trial, id):

        # Find the relationship between the torso and the ground frame
        T = self.find_transform("odom_combined", "torso_lift_link")

        # Define the spawning location of the object with respect to the torso of the robot
        x = self.test_x[trial]
        y = self.test_y[trial]
        z = -((T[2, -1] - TABLE_HEIGHT)-self.test_names[id].get('h')/2)

        d = [x, y, z, 1]

        # Find the location of the object with respect to the ground frame
        pw = np.dot(T, d)
        pw = pw[0:3]

        # Define the final transformation
        yaw = self.test_yaw[trial]

        orient = transform.transformations.quaternion_from_euler(0, 0, yaw)
        pos = [pw[0]*CONVERT, -pw[1]*CONVERT, pw[2]*CONVERT+SPAWN]
        self.move_object(self.test_names[id].get('name'), pos, orient)

        return [x, y, z], math.pi-yaw, T

    def move_testing_object(self, x, y, id):

        # Find the relationship between the torso and the ground frame
        T = self.find_transform("odom_combined", "torso_lift_link")

        # Define the spawning location of the object with respect to the torso of the robot
        z = -((T[2, -1] - TABLE_HEIGHT)-self.test_names[id].get('h')/2)

        d = [x, y, z, 1]

        # Find the location of the object with respect to the ground frame
        pw = np.dot(T, d)
        pw = pw[0:3]

        # Define the final transformation
        yaw = 0

        orient = transform.transformations.quaternion_from_euler(0, 0, yaw)
        pos = [pw[0]*CONVERT, -pw[1]*CONVERT, pw[2]*CONVERT+SPAWN]
        self.move_object(self.test_names[id].get('name'), pos, orient)

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

            o = transform.transformations.quaternion_from_euler(0, 1.571, yaw)

        if b is 'Side':

            offset = self.names[i].get('h')/2
            low = pos[2]+PAD/2
            high = pos[2]+offset/2

            k = np.zeros((self.key_point_length, 3))
            zeta = np.linspace(low, high, self.key_point_length)

            for i in range(0, self.key_point_length):
                k[i] = [pos[0], pos[1]-offset, zeta[i]]

            o = transform.transformations.quaternion_from_euler(0, 0, 1.571)

        return [k, o]

    def generate_test_key_points(self, i, pos, yaw):

        b = self.pick_test_behavior(i)
        print(b)

        if b is 'Top':

            offset = self.names[i].get('h')/2
            low = pos[2]+offset+PAD
            high = pos[2]+offset+2*PAD

            k = np.zeros((self.key_point_length, 3))
            zeta = np.linspace(low, high, self.key_point_length)

            for i in range(0, self.key_point_length):
                k[i] = [pos[0], pos[1], zeta[i]]

            o = transform.transformations.quaternion_from_euler(0, 1.571, yaw)

        if b is 'Side':

            offset = self.names[i].get('h')/2
            low = pos[2]+PAD/2
            high = pos[2]+offset/2

            k = np.zeros((self.key_point_length, 3))
            zeta = np.linspace(low, high, self.key_point_length)

            for i in range(0, self.key_point_length):
                k[i] = [pos[0], pos[1]-offset, zeta[i]]

            o = transform.transformations.quaternion_from_euler(0, 0, 1.571)

        return [k, o]

    def move_test_object_back(self, i):

        pos = [self.test_names[i].get('x'), self.test_names[i].get('y'), self.test_names[i].get('z')]
        orient = [0, 0, 0, 1]

        self.move_object(self.test_names[i].get('name'), pos, orient)

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


