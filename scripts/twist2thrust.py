#!/usr/bin/python3
import rospy
import numpy as np
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

#max u=2 m/s max w= 0.5 rad/s

class VelNode:
    def __init__(self):
        rospy.init_node('vel_node_py', anonymous=True)
        self.subscriber = rospy.Subscriber('boat/cmd_vel', Twist, self.callback)
        self.pub_l = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.pub_r = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=10)
        self.pub_l1 = rospy.Publisher('/wamv/thrusters/left1_thrust_cmd', Float32, queue_size=10)
        self.pub_r1 = rospy.Publisher('/wamv/thrusters/right1_thrust_cmd', Float32, queue_size=10)
      
        self.m_l = Float32()
        self.m_r = Float32()
        self.rate = rospy.Rate(10)  # Frecuencia de ejecución de 10 Hz

    def callback(self, data):
        # Realiza aquí las operaciones o acciones deseadas con los datos recibidos

        DISTANCIA_ENTRE_PROPULSORES=0.7*2
        matrizn=[[0], [0]]

        matrizn[0]=data.linear.x/2 + data.angular.z * DISTANCIA_ENTRE_PROPULSORES / (2*0.49)
        if abs(matrizn[0])>1:
            matrizn[0]=math.copysign(1, matrizn[0])*1
        matrizn[1]=data.linear.x/2 - data.angular.z * DISTANCIA_ENTRE_PROPULSORES / (2*0.49)
        if abs(matrizn[1])>1:
            matrizn[1]=math.copysign(1, matrizn[1])*1


        self.m_l.data=matrizn[1]
        self.m_r.data=matrizn[0]
        self.pub_l.publish(self.m_l)
        self.pub_r.publish(self.m_r)
        self.pub_l1.publish(self.m_l)
        self.pub_r1.publish(self.m_r)
        self.rate.sleep()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    vel_node = VelNode()
    vel_node.run()
