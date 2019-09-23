#!/usr/bin/env python

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# Directory
directory_dmp = '/home/michail/Desktop/Bremen/catkin_ws/src/Data/Unreal-Dataset/Final_Results_dmp_3'
directory_vel = '/home/michail/Desktop/Bremen/catkin_ws/src/Data/Unreal-Dataset/Final_Results_vel_3'

# Distance vectors for dmp results
dmp_pos_0 = np.loadtxt(directory_dmp+'/position_0.txt')
dmp_pos_1 = np.loadtxt(directory_dmp+'/position_1.txt')
dmp_pos_2 = np.loadtxt(directory_dmp+'/position_2.txt')
dmp_pos_3 = np.loadtxt(directory_dmp+'/position_3.txt')


# Distance vectors for dmp results
vel_pos_0 = np.loadtxt(directory_vel+'/position_0.txt')
vel_pos_1 = np.loadtxt(directory_vel+'/position_1.txt')
vel_pos_2 = np.loadtxt(directory_vel+'/position_2.txt')
vel_pos_3 = np.loadtxt(directory_vel+'/position_3.txt')

# position of the object
o_0 = np.array([[0.627, 0.183, -0.3329982452873298], [0.627, 0.183, -0.3229982452873298]])
o_1 = np.array([[0.682, -0.029, -0.3313979920868596], [0.682, -0.029, -0.3213979920868596]])
o_2 = np.array([[0.488, 0.147, -0.34344763439947135], [0.488, 0.147, -0.33344763439947135]])
o_3 = np.array([[0.571, 0.038, -0.36928189657163096], [0.571, 0.038, -0.35928189657163096]])

# Plot position for object 0
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vel_pos_0[:, 0], vel_pos_0[:, 1], vel_pos_0[:, 2],color='g', label='velocity')
ax.plot(dmp_pos_0[:, 0], dmp_pos_0[:, 1], dmp_pos_0[:, 2],color='b', label='dmp')
ax.plot([o_0[0, 0]], [o_0[0, 1]], [o_0[0, 2]], 'o',color='r',label='object')
ax.legend()

ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.view_init(elev=32., azim=132)

ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)

plt.tick_params(axis='both', which='minor', labelsize=5)
plt.grid(True)
plt.show()


# Plot position for object 1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vel_pos_1[:, 0], vel_pos_1[:, 1], vel_pos_1[:, 2],color='g', label='velocity')
ax.plot(dmp_pos_1[:, 0], dmp_pos_1[:, 1], dmp_pos_1[:, 2],color='b', label='dmp')
ax.plot([o_1[0, 0]], [o_1[0, 1]], [o_1[0, 2]],'o',color='r',label='object')
ax.legend()

ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.view_init(elev=32., azim=132)

ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)

plt.tick_params(axis='both', which='minor', labelsize=5)
plt.grid(True)
plt.show()

# Plot position for object 2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vel_pos_2[:, 0], vel_pos_2[:, 1], vel_pos_2[:, 2],color='g', label='velocity')
ax.plot(dmp_pos_2[:, 0], dmp_pos_2[:, 1], dmp_pos_2[:, 2],color='b', label='dmp')
ax.plot([o_2[0, 0]], [o_2[0, 1]], [o_2[0, 2]], 'o',color='r',label='object')
ax.legend()

ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.view_init(elev=32., azim=132)

ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)

plt.tick_params(axis='both', which='minor', labelsize=5)
plt.grid(True)
plt.show()

# Plot position for object 3
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(vel_pos_3[:, 0], vel_pos_3[:, 1], vel_pos_3[:, 2],color='g', label='velocity')
ax.plot(dmp_pos_3[:, 0], dmp_pos_3[:, 1], dmp_pos_3[:, 2],color='b', label='dmp')
ax.plot([o_3[0, 0]], [o_3[0, 1]], [o_3[0, 2]], 'o',color='r',label='object')
ax.legend()

ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.view_init(elev=32., azim=132)

ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)

plt.tick_params(axis='both', which='minor', labelsize=5)
plt.grid(True)
plt.show()

















