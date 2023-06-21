#!/usr/bin/python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class VelNode:
    def __init__(self):
        rospy.init_node('vel_node_py', anonymous=True)
        self.subscriber = rospy.Subscriber('boat/cmd_vel', Twist, self.callback)
        self.pub_l = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.pub_r = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=10)
        self.m_l = Float32()
        self.m_r = Float32()
        self.rate = rospy.Rate(100)  # Frecuencia de ejecución de 10 Hz

    def normalize_value(self, x, v_min, v_max):
        x_normalized = (2 * (x - v_min)) / (v_max - v_min) - 1
        return x_normalized

    def callback(self, data):
        # Realiza aquí las operaciones o acciones deseadas con los datos recibidos

        DISTANCIA_ENTRE_RUEDAS=0.647*2
        matrizn=[[0], [0]]
        matrizn[0]=(2 * data.linear.x - data.angular.z * DISTANCIA_ENTRE_RUEDAS) / 2
        matrizn[1]=(2 * data.linear.x + data.angular.z * DISTANCIA_ENTRE_RUEDAS) / 2

        self.m_l.data=matrizn[0]
        self.m_r.data=matrizn[1]
        self.pub_l.publish(self.m_l)
        self.pub_r.publish(self.m_r)
        self.rate.sleep()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    vel_node = VelNode()
    vel_node.run()
