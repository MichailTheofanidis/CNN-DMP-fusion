#!/usr/bin/env python
import roslib
import rospy
import tf2_ros
import tf
import numpy as np
import random as rn
roslib.load_manifest('service_pkg')

rospy.init_node('tf_listener')

tf_buffer = tf2_ros.Buffer()
tf2_listener = tf2_ros.TransformListener(tf_buffer)

# Define the transformation from the robot to the object
x = round(rn.uniform(0.4, 0.65), 3)
y = round(rn.uniform(-0.33, 0.33), 3)
z = -0.28

d = [x, y, z, 1]

print(d)
print('--------')

# Find the transformation from the robot world to the base of the robot
t = tf_buffer.lookup_transform("odom_combined", "torso_lift_link", rospy.Time(0), rospy.Duration(10))
print(t)
print('--------')

p = [t.transform.translation.x,
     t.transform.translation.y,
     t.transform.translation.z]

q = [t.transform.rotation.x,
     t.transform.rotation.y,
     t.transform.rotation.z,
     t.transform.rotation.w]

Tr = np.array(tf.transformations.quaternion_matrix([q[0], q[1], q[2], q[3]]))
Tr[0:3, -1] = p

print(Tr)
print('--------')
# Find the transformation from the object to the robot world
pw = np.dot(Tr, d)
pw = pw[0:3]

print(pw)
print('--------')




