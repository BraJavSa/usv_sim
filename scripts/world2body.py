#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

import math
import numpy as np

class RobotController:
    def __init__(self):
        rospy.init_node('world2body_node', anonymous=True)
        self.subscriber = rospy.Subscriber('/wamv/sensors/position/p3d_wamv', Odometry, self.odometry_callback)
        self.odom_pub= rospy.Publisher("boat/odom", Odometry, queue_size=10)
        self.odom = Odometry()
        rate = rospy.Rate(10)

    def odometry_callback(self, data):
        chasisPos_X = data.pose.pose.position.x
        chasisPos_Y = data.pose.pose.position.y
        chasisPos_Z = data.pose.pose.position.z
        quaterW = data.pose.pose.orientation.w
        quaterX = data.pose.pose.orientation.x 
        quaterY = data.pose.pose.orientation.y
        quaterZ = data.pose.pose.orientation.z 
        orientation_list = [quaterX, quaterY, quaterZ, quaterW]
        euler = euler_from_quaternion(orientation_list)
        orient = euler[2]
        signo = 1
        if orient < 0:
            signo = -1

        orient = -signo * (math.pi - signo * orient)
        chasisPos = [chasisPos_X, chasisPos_Y, chasisPos_Z]

        chasisVel_W_X = data.twist.twist.linear.x 
        chasisVel_W_Y = data.twist.twist.linear.y
        chasisVel_W_w = data.twist.twist.angular.z
        vels = [chasisVel_W_X, chasisVel_W_Y, chasisVel_W_w]
        velsArr = np.array(vels)
        a=0.1
        JacChasis = [[math.cos(orient), -math.sin(orient), -a*math.sin(a+orient)],
                     [math.sin(orient), math.cos(orient), a*math.cos(a+orient)], [0, 0, 1]]
        invJac = np.linalg.inv(JacChasis)
        chasisVel = invJac.dot(velsArr)
        chasisVel[0] = -chasisVel[0]
        

        # Imprimir las velocidades calculadas
        #print("Velocity")
        self.odom.twist.twist.linear.x=chasisVel[0]
        self.odom.twist.twist.linear.y=chasisVel[1]
        self.odom.twist.twist.linear.z=chasisVel[2]
        self.odom.twist.twist.angular.x=data.twist.twist.angular.x 
        self.odom.twist.twist.angular.y=data.twist.twist.angular.y 
        self.odom.twist.twist.angular.z=data.twist.twist.angular.z 
        self.odom.pose.pose.position.x = data.pose.pose.position.x
        self.odom.pose.pose.position.y = data.pose.pose.position.y
        self.odom.pose.pose.position.z = data.pose.pose.position.z
        self.odom.pose.pose.orientation.x = euler[2]
        self.odom.pose.pose.orientation.y = euler[1]
        self.odom.pose.pose.orientation.z = euler[0]
        self.odom_pub.publish(self.odom)
        #print(chasisVel)


    def start(self):
        rospy.spin()

if __name__ == '__main__':
    controller = RobotController()
    controller.start()
