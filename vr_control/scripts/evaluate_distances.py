#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Directory
directory_dmp = '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Results_dmp_3'
directory_vel = '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Results_vel_3'

# Distance vectors for dmp results
dmp_distance_0 = np.loadtxt(directory_dmp+'/distances_0.txt')
dmp_distance_1 = np.loadtxt(directory_dmp+'/distances_1.txt')
dmp_distance_2 = np.loadtxt(directory_dmp+'/distances_2.txt')
dmp_distance_3 = np.loadtxt(directory_dmp+'/distances_3.txt')

# Distance vectors for dmp results
vel_distance_0 = np.loadtxt(directory_vel+'/distances_0.txt')
vel_distance_1 = np.loadtxt(directory_vel+'/distances_1.txt')
vel_distance_2 = np.loadtxt(directory_vel+'/distances_2.txt')
vel_distance_3 = np.loadtxt(directory_vel+'/distances_3.txt')

# Average distance vector for dmp resuls
dmp_avg_0 = np.sum(dmp_distance_0)/len(dmp_distance_0)
dmp_avg_1 = np.sum(dmp_distance_1)/len(dmp_distance_1)
dmp_avg_2 = np.sum(dmp_distance_2)/len(dmp_distance_2)
dmp_avg_3 = np.sum(dmp_distance_3)/len(dmp_distance_3)

# Average distance vector for vel resuls
vel_avg_0 = np.sum(vel_distance_0)/len(vel_distance_0)
vel_avg_1 = np.sum(vel_distance_1)/len(vel_distance_1)
vel_avg_2 = np.sum(vel_distance_2)/len(vel_distance_2)
vel_avg_3 = np.sum(vel_distance_3)/len(vel_distance_3)


# Plot histograms
labels = ['Object 1', 'Object 2', 'Object 3', 'Object 4']
dmp_means = np.round([dmp_avg_0, dmp_avg_1, dmp_avg_2, dmp_avg_3], 3)
vel_means = np.round([vel_avg_0, vel_avg_1, vel_avg_2, vel_avg_3], 3)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, vel_means, width, label='velocity')
rects2 = ax.bar(x + width/2, dmp_means, width, label='dmp parameters')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Distance (m)')
ax.set_title('Error between end effector and object')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()