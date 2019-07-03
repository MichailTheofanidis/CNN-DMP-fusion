#!/usr/bin/env python

import os
import cv2
import rospy
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from service_pkg.srv import *


class Recorder(threading.Thread):

    def __init__(self, directory_id, ):

        super(Recorder, self).__init__()
        self.directory_id = directory_id
        self.trajectory_id = 0
        self.storing_directory = '/home/mtheofanidis/Desktop/Data/Unreal-Dataset/Dataset_'
        self.counter = 0
        self.daemon = True
        self.paused = True
        self.state = threading.Condition()

        rospy.wait_for_service("return_camera_image")

    def make_directory(self):

        # Create the directory to store the data
        directory = self.storing_directory + str(self.directory_id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def make_image_directory(self):

        current_dir = self.make_directory()

        img_dir = current_dir + '/images_' + str(self.trajectory_id)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

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
            #cv2.imshow("Image window", cv_image)

            img_dir = self.make_image_directory()
            cv2.imwrite(img_dir + '/shot_' + str(self.counter) + '.jpg', cv_image)
            self.counter += 1

            cv2.waitKey(1)

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.

    def pause(self):
        with self.state:
            self.paused = True  # Block self.






