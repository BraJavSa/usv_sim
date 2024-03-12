#!/usr/bin/python3
import rospy
import numpy as np
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
        self.rate = rospy.Rate(100)  # Frecuencia de ejecución de 10 Hz

    def callback(self, data):
        # Realiza aquí las operaciones o acciones deseadas con los datos recibidos

        DISTANCIA_ENTRE_RUEDAS=0.647*2
        matrizn=[[0], [0]]
        if abs(data.linear.x)<=2:
            vx=data.linear.x/2
        elif data.linear.x < -2:
            vx=-1
        else:
            vx=1
        if abs(data.angular.z)<=0.5:
            vy=data.angular.z
        elif data.angular.z < -0.5:
            vy=-0.5
        else:
            vy=0.5
        vy=vy*3.0915

        
        matrizn[0]=(2 * vx - vy * DISTANCIA_ENTRE_RUEDAS) / 2
        matrizn[1]=(2 * vx + vy * DISTANCIA_ENTRE_RUEDAS) / 2

        self.m_l.data=matrizn[0]
        self.m_r.data=matrizn[1]
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
