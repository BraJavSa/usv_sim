#!/usr/bin/python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
#max u=2 m/s max w= 0.5 rad/s
KP = 1.05  # Ganancia proporcional
KD = 0.05  # Ganancia derivativa
KPa = 1.0  # Ganancia proporcional
KDa = 0.01  # Ganancia derivativa
class VelNode:
    def __init__(self):
        rospy.init_node('vel_node_py', anonymous=True)
        self.subscriber = rospy.Subscriber('boat/cmd_vel', Twist, self.cmd_vel_callback)
        self.subscriber_odom = rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)
        self.pub_l = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.pub_r = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=10)
        self.pub_l1 = rospy.Publisher('/wamv/thrusters/left1_thrust_cmd', Float32, queue_size=10)
        self.pub_r1 = rospy.Publisher('/wamv/thrusters/right1_thrust_cmd', Float32, queue_size=10)    
        self.m_l = Float32()
        self.m_r = Float32()
        self.desired_linear_velocity = 0.0
        self.desired_angular_velocity = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.error_integral_linear = 0.0
        self.error_integral_angular = 0.0
        self.last_error_linear = 0.0
        self.last_error_angular = 0.0

        self.rate = rospy.Rate(100)  # Frecuencia de ejecución de 10 Hz

    def cmd_vel_callback(self, data):
        self.desired_linear_velocity = data.linear.x
        self.desired_angular_velocity = data.angular.z

    def calculate_control_output(self):
        error_linear = self.desired_linear_velocity - self.current_linear_velocity
        error_angular = self.desired_angular_velocity - self.current_angular_velocity
        self.control_linear = KP * error_linear 
        self.control_angular = KP * error_angular 
        self.error_integral_linear += error_linear
        self.error_integral_angular += error_angular

        self.control_linear = KP * error_linear + KD * (error_linear - self.last_error_linear)
        self.control_angular = KPa * error_angular  + KDa * (error_angular - self.last_error_angular)

        self.last_error_linear = error_linear
        self.last_error_angular = error_angular


    def odom_callback(self, data):
        self.current_linear_velocity = data.twist.twist.linear.x
        self.current_angular_velocity = data.twist.twist.angular.z

    def tw2thcallback(self):
        # Realiza aquí las operaciones o acciones deseadas con los datos recibidos
        DISTANCIA_ENTRE_PROPULSORES=0.7*2
        matrizn=[[0], [0]]

        matrizn[0]=(self.desired_linear_velocity/2 + self.desired_angular_velocity * DISTANCIA_ENTRE_PROPULSORES / (2*0.49))+(self.control_linear/2 + self.control_angular * DISTANCIA_ENTRE_PROPULSORES / (2*0.49))
        if abs(matrizn[0])>1:
            matrizn[0]=math.copysign(1, matrizn[0])*1
        matrizn[1]=(self.desired_linear_velocity/2 - self.desired_angular_velocity * DISTANCIA_ENTRE_PROPULSORES / (2*0.49))+(self.control_linear/2 - self.control_angular * DISTANCIA_ENTRE_PROPULSORES / (2*0.49))
        if abs(matrizn[1])>1:
            matrizn[1]=math.copysign(1, matrizn[1])*1
        self.m_l.data=matrizn[1]
        self.m_r.data=matrizn[0]
        self.pub_l.publish(self.m_l)
        self.pub_r.publish(self.m_r)
        self.pub_l1.publish(self.m_l)
        self.pub_r1.publish(self.m_r)

    def run(self):
        while not rospy.is_shutdown():
            self.calculate_control_output()
            self.tw2thcallback()
            self.rate.sleep()

if __name__ == '__main__':
    vel_node = VelNode()
    vel_node.run()
