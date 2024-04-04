#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

from tf.transformations import euler_from_quaternion
from setpoint_gazebo_delete import delete_gazebo_model
from setpoint_gazebo_print import  load_gazebo_model


class PositionController:
    def __init__(self):
        self.hxd = 0
        self.hyd = 0
        self.uRef=0
        self.wRef=0
        self.hxe=0
        self.hye=0
        self.a=0.1 #punto de interes
        self.current_x=0
        self.current_y=0
        self.current_z=0
        self.ori_x=0
        self.ori_y=0
        self.ori_z=0
        self.ori_w=0
        #kv= u max,    wmax= kk*pi + kv*0.5
        self.rate = rospy.Rate(50)
        self.kv=1.8
        self.kk=1.5
        self.k2=0.08
        self.odom_subscriber = rospy.Subscriber('wamv/sensors/position/p3d_wamv', Odometry, self.odom_callback)
        self.pose_sub = rospy.Subscriber("/boat/pose_d", PoseStamped, self.update_position)
        self.vel_publisher = rospy.Publisher('/boat/cmd', Twist, queue_size=10)


    def odom_callback(self, odom_msg):
        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.current_z = odom_msg.pose.pose.position.z
        self.ori_x = odom_msg.pose.pose.orientation.x
        self.ori_y = odom_msg.pose.pose.orientation.y
        self.ori_z = odom_msg.pose.pose.orientation.z
        self.ori_w = odom_msg.pose.pose.orientation.w

    def update_position(self, data):
        self.hxd = data.pose.position.x
        self.hyd = data.pose.position.y

            

    
    def normal_angulo(self, angulo):
        if angulo>3.14159:
            angulo=angulo-2*3.14159
        elif angulo<-3.14159:
            angulo=angulo+2*3.14159
        else:
            pass
        return(angulo)
            

    def control(self):
        self.hxe=self.hxd-self.current_x        
        self.hye=self.hyd-self.current_y
        
        error=math.sqrt(self.hxe**2 + self.hye**2)

        
        a_e= math.atan2(self.hye, self.hxe)
        orientation_list = [self.ori_x, self.ori_y, self.ori_z, self.ori_w]
        current_angle = euler_from_quaternion(orientation_list)
        

        # Calcular la diferencia de ángulo entre el ángulo deseado y el ángulo actual
        angular_difference = self.normal_angulo(self.normal_angulo(a_e) - self.normal_angulo(current_angle[2]))
        
        if error <=0.5:
            self.uRef=0
            self.wRef=0          
        else:
            self.uRef=self.kv*math.tanh(self.k2*error)*math.cos(angular_difference)
            self.wRef=self.kk*angular_difference+self.kv*(math.tanh(self.k2*error)/error)*math.sin(angular_difference)*math.cos(angular_difference)
        


        twist_msg = Twist()
        twist_msg.linear.x = self.uRef
        if abs(self.wRef)>0.7:
            self.wRef==math.copysign(1, self.wRef)*0.7
        twist_msg.angular.z = self.wRef
        self.vel_publisher.publish(twist_msg)
        self.rate.sleep()
        


# Inicialización de ROS
rospy.init_node('position_controller')

# Ejemplo de uso
controller = PositionController()

while not rospy.is_shutdown():
    controller.control()