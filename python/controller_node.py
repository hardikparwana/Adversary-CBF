#!/usr/bin/env python

import rospy
import roslib
import numpy as np
import cvxpy as cp

from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion

rover7 = []
rover3 = []
cf1 = []
cf2 = []
cf3 = []
cf4 = []

def rover7poseCallback(data):
    global rover7
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    rover7 = [data.transform.translation.x, data.transform.translation.y, yaw  ]

def rover3poseCallback(data):
    global rover3
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    rover3 = [data.transform.translation.x, data.transform.translation.y, yaw  ]
    
def cf1poseCallback(data):
    global cf1
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    cf1 = [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z  ]
    
def cf2poseCallback(data):
    global cf2
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    cf2 = [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z  ]
    
def cf3poseCallback(data):
    global cf3
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    cf3 = [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z  ]
    
def cf4poseCallback(data):
    global cf4
    quaternion = data.transform.rotation
    quat_tf = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    yaw = euler_from_quaternion(quat_tf)[2]
    cf4 = [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z  ]

def controller():
    rospy.init_node('controlelr_node', anonymous=True)  
    rate = rospy.Rate(20) # 10hz


    pathPub = rospy.Publisher("/path",Path,queue_size=1)

    rover7PoseSub = rospy.Subscriber("/vicon/rover7/rover7",TransformStamped,rover7poseCallback)
    rover3PoseSub = rospy.Subscriber("/vicon/rover3/rover3",TransformStamped,rover3poseCallback)
    cf1PoseSub = rospy.Subscriber("/vicon/cf1/cf1",TransformStamped,cf1poseCallback)
    cf2PoseSub = rospy.Subscriber("/vicon/cf2/cf2",TransformStamped,cf2poseCallback)
    cf3PoseSub = rospy.Subscriber("/vicon/cf3/cf3",TransformStamped,cf3poseCallback)
    cf4PoseSub = rospy.Subscriber("/vicon/cf4/cf4",TransformStamped,cf4poseCallback)



if __name__ == "__main__":
    
    try:
        controller()
    except rospy.ROSInterruptException:
        pass 
