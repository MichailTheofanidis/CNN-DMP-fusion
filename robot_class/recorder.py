#!/usr/bin/env python

import glob
import imageio
import os
import cv2
import rospy
import threading
import pickle
import numpy as np

from natsort import natsorted
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from service_pkg.srv import *

demoX = []
demoU = []
demoDMP = []
distances = []
state = []
state_ee = []


class Recorder(threading.Thread):

    def __init__(self, directory_id, ):

        super(Recorder, self).__init__()
        self.directory_id = directory_id
        self.storing_directory = "/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_"
        self.image_directory = ''
        self.counter = 0
        self.daemon = True
        self.paused = True
        self.state = threading.Condition()
        self.interpolate = 400

        rospy.wait_for_service("return_camera_image")

    def make_directory(self):

        # Create the directory to store the data
        directory = self.storing_directory + str(self.directory_id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def make_image_directory(self, id):

        current_dir = self.make_directory()

        img_dir = current_dir + '/object_' + str(id)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        self.image_directory = img_dir

        return img_dir

    def run(self):
        self.resume()
        while True:
            with self.state:
                if self.paused:
                    self.state.wait()

            s = rospy.ServiceProxy("return_camera_image", ReturnImages)

            img_data = Image()

            img_data.width = s([]).width
            img_data.height = s([]).height
            img_data.encoding = s([]).encoding
            img_data.data = s([]).data

            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(img_data, "passthrough")
            resized = cv2.resize(cv_image, (125, 125), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Image window", resized)

            cv2.imwrite(self.image_directory + '/shot_' + str(self.counter) + '.png', resized)
            self.counter += 1

            cv2.waitKey(1)

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.

    def pause(self):
        with self.state:
            self.paused = True  # Block self.

    def make_gif(self, runs):

        # Collect the images
        file_names = natsorted(glob.glob(self.image_directory+'/*.png'))

        # Interpolate images
        file_index = np.floor(np.linspace(1, len(file_names)-1, self.interpolate))

        # Make an array of images
        images = []
        for indx in file_index:
            images.append(imageio.imread(file_names[int(indx)]))

        #for indx in file_names:
            #images.append(imageio.imread(indx))

        # Store the images as a gif
        imageio.mimsave(self.image_directory+'/demo_' + str(runs) + '.gif', images, fps=30, palettesize=256, format='GIF-FI')

        # Remove the generated images
        for f in file_names:
            os.remove(f)

        # Wait for the data processing
        rospy.sleep(rospy.Duration(1))

    def store_angles(self, traj, d_traj, dmp):
        global demoX, demoU, demoDMP

        demoX.append(traj)
        demoU.append(d_traj)
        demoDMP.append(dmp)

    def store_distances(self, distance):
        global distances

        distances.append(distance)

    def store_trajectories(self, directory,q,p,id):
        global state, state_ee

        save_directory_state = directory + '/joints_' + str(id) + '.txt'
        save_directory_state_ee = directory + '/position_' + str(id) + '.txt'

        np.savetxt(save_directory_state, q)
        np.savetxt(save_directory_state_ee, p)

    def store_performance(self, directory, id):
        global distances

        save_directory = directory + '/distances_' + str(id) + '.txt'

        np.savetxt(save_directory, distances)

    def store_demo(self, id):
        global demoX, demoU, demoDMP

        # Directory
        dir = self.make_directory()
        final_dir = dir + '/demos_' + str(id) + '.pkl'

        # Define the demo data
        demo = {'demoX': np.array(demoX), 'demoU': np.array(demoU), 'demoDMP': np.array(demoDMP)}

        # Store the data
        pickle_out = open(final_dir, "wb")
        pickle.dump(demo, pickle_out)
        pickle_out.close()

    def empty_array(self):
        global demoX, demoU, demoDMP

        # Empty the storing arrays
        demoX = []
        demoU = []
        demoDMP = []

    def empty_distance(self):
        global distances

        # Empty the array which stores the distances
        distances = []



