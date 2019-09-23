#!/usr/bin/env python

import roslib
import rospy
from service_pkg.srv import *
from sensor_msgs.msg import Image
import threading
roslib.load_manifest('service_pkg')


# holds the latest states obtained from joint_states messages
class CameraListener:

    def __init__(self):
        rospy.init_node('camera_listener')
        self.lock = threading.Lock()
        self.height = []
        self.width = []
        self.encoding = []
        self.data = []
        self.thread = threading.Thread(target=self.camera_listener)
        self.thread.start()

        s = rospy.Service('return_camera_image', ReturnImages, self.return_images)

    # thread function: listen for image messages
    def camera_listener(self):
        rospy.Subscriber('/unreal_vison/image_color', Image, self.camera_listener_callback)
        rospy.spin()

    # callback function: when an image arrive save its values
    def camera_listener_callback(self, msg):
        self.lock.acquire()
        self.height = msg.height
        self.width = msg.width
        self.encoding = msg.encoding
        self.data = msg.data
        self.lock.release()

    # server callback: returns image data
    def return_images(self, dummy):

        height = self.height
        width = self.width
        encoding = self.encoding
        data = self.data

        return ReturnImagesResponse(height, width, encoding, data)


# run the server
if __name__ == "__main__":

    CameraListener()

    print "camera_listener server started, waiting for queries"
    rospy.spin()
