#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped

from setpoint_gazebo_delete import delete_gazebo_model
from setpoint_gazebo_print import  load_gazebo_model


class PositionController:
    def __init__(self):
        self.hxd = 0
        self.hyd = 0
        self.hxd1 = 0
        self.hyd1 = 0
        self.bandera=False
        #kv= u max,    wmax= kk*pi + kv*0.5
        self.rate = rospy.Rate(1)
        self.pose_sub = rospy.Subscriber("/boat/pose_d", PoseStamped, self.update_position)
   
    def update_position(self, data):
        self.hxd = data.pose.position.x
        self.hyd = data.pose.position.y
        if self.hxd1 == self.hxd and self.hyd1 == self.hyd:
            pass
        else:
            self.hxd1 = self.hxd
            self.hyd1 = self.hyd
            if self.bandera==True:
                delete_gazebo_model("win_point")
                load_gazebo_model(self.hxd,self.hyd)
                
            else:
                load_gazebo_model(self.hxd,self.hyd)
                self.bandera=True
            

            

    def control(self):
        self.rate.sleep()


# Inicialización de ROS
rospy.init_node('set_desired_point')
controller = PositionController()

while not rospy.is_shutdown():
    controller.control()