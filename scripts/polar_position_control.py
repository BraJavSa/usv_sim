#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class PositionController:
    def __init__(self):
        self.hxd = 100
        self.hyd = 100
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

        self.kv=1.9

        self.kk=0.5
        self.k2=1
        self.odom_subscriber = rospy.Subscriber('wamv/sensors/position/p3d_wamv', Odometry, self.odom_callback)
        self.vel_publisher = rospy.Publisher('/boat/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(100)

    def odom_callback(self, odom_msg):
        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.current_z = odom_msg.pose.pose.position.z
        self.ori_x = odom_msg.pose.pose.orientation.x
        self.ori_y = odom_msg.pose.pose.orientation.y
        self.ori_z = odom_msg.pose.pose.orientation.z
        self.ori_w = odom_msg.pose.pose.orientation.w

    
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
        
        if abs(angular_difference)>0.3 and error>=10:
            
            self.wRef=self.kk*angular_difference+self.kv*(math.tanh(self.k2*error)/error)*math.sin(angular_difference)*math.cos(angular_difference)
            self.uRef=0
        else:
            self.kk=0.5
            if error <=2:
                self.uRef=0
                self.wRef=0          
            else:
                self.uRef=self.kv*math.tanh(self.k2*error)*math.cos(angular_difference)
                self.wRef=self.kk*angular_difference+self.kv*(math.tanh(self.k2*error)/error)*math.sin(angular_difference)*math.cos(angular_difference)
            


        twist_msg = Twist()
        twist_msg.linear.x = self.uRef
        twist_msg.angular.z = self.wRef
        self.vel_publisher.publish(twist_msg)
        self.rate.sleep()
        


# Inicialización de ROS
rospy.init_node('position_controller')

# Ejemplo de uso
controller = PositionController()

while not rospy.is_shutdown():
    controller.control()