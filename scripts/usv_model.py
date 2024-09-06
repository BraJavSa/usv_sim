#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np

class USVModel:
    def __init__(self):
        # Estado inicial: [u, v, r, x, y, psi]
        self.state = np.zeros(6)  # [u, v, r, x, y, psi]
        self.perturbation = np.zeros(2)  # Perturbaciones iniciales [viento, oleaje]
        self.control_action = Twist()  # Velocidades iniciales en cero

        # Inicializar la posición en (0.0, 0.0, 0.0) y orientación común
        self.initial_pose = Pose()
        self.initial_pose.position.x = 158
        self.initial_pose.position.y = 108
        self.initial_pose.position.z = 0.0
        self.initial_pose.orientation.z = 0.0  # Orientación común (en radianes)
        self.initial_pose.orientation.w = 1.0  # Cuaternión de identidad

        rospy.Subscriber('/usv/control_action', Twist, self.control_callback)
        rospy.Subscriber('/usv/perturbation', Twist, self.perturbation_callback)
        self.pose_pub = rospy.Publisher('/usv/pose', Pose, queue_size=10)

        self.rate = rospy.Rate(100)  

    def control_callback(self, msg):
        self.control_action = msg

    def perturbation_callback(self, msg):
        self.perturbation[0] = msg.linear.x  # Viento
        self.perturbation[1] = msg.angular.z  # Oleaje

    def update_state(self):
        # Implementación simplificada del modelo dinámico de Fossen
        u, v, r = self.state[0], self.state[1], self.state[2]
        x, y, psi = self.state[3], self.state[4], self.state[5]

        # Actualizar el estado con las velocidades y perturbaciones
        u_dot = self.control_action.linear.x + self.perturbation[0]
        r_dot = self.control_action.angular.z + self.perturbation[1]

        # Actualizar las velocidades
        u += u_dot * 0.2  # dt = 200 ms
        r += r_dot * 0.2

        # Actualizar las posiciones
        x += u * np.cos(psi) * 0.2
        y += u * np.sin(psi) * 0.2
        psi += r * 0.2

        self.state = np.array([u, v, r, x, y, psi])

    def publish_pose(self):
        pose = Pose()
        pose.position.x = self.state[3]
        pose.position.y = self.state[4]
        pose.position.z = self.initial_pose.position.z

        # Convertir el ángulo psi a cuaterniones para la orientación
        pose.orientation.z = np.sin(self.state[5] / 2.0)
        pose.orientation.w = np.cos(self.state[5] / 2.0)

        self.pose_pub.publish(pose)

    def run(self):
        # Inicializar el estado en cero
        self.state = np.zeros(6)  # Todas las velocidades y posiciones en cero

        while not rospy.is_shutdown():
            self.update_state()
            self.publish_pose()
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('usv_model', anonymous=True)
    usv_model = USVModel()
    usv_model.run()

