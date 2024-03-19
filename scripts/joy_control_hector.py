#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyTeleop:
    def __init__(self):
        rospy.init_node('joy_teleop')

        # Suscribirse al topic del joystick
        rospy.Subscriber("/joy2", Joy, self.joy_callback)

        # Publicar al topic de comando twist
        self.cmd_pub = rospy.Publisher('/command/twist', Twist, queue_size=10)

        # Velocidades máximas permitidas
        self.max_linear_x_speed = 1.0  # Ajustar según sea necesario
        self.max_linear_y_speed = 1.0  # Ajustar según sea necesario
        self.max_linear_z_speed = 1.0  # Ajustar según sea necesario
        self.max_angular_z_speed = 1.0  # Ajustar según sea necesario

    def joy_callback(self, joy_msg):
        # Mapear los valores del joystick a velocidades lineales y angulares
        linear_x_speed = self.max_linear_x_speed * joy_msg.axes[1]  # eje vertical izquierdo (arriba/abajo)
        linear_y_speed = self.max_linear_y_speed * joy_msg.axes[0]  # eje horizontal izquierdo (izquierda/derecha)
        linear_z_speed = self.max_linear_z_speed * joy_msg.axes[4]  # botón RT (adelante/atrás)
        angular_z_speed = self.max_angular_z_speed * joy_msg.axes[3]  # eje horizontal derecho (izquierda/derecha)

        # Crear el mensaje Twist y publicarlo
        twist_msg = Twist()
        twist_msg.linear.x = linear_x_speed
        twist_msg.linear.y = linear_y_speed
        twist_msg.linear.z = linear_z_speed
        twist_msg.angular.z = angular_z_speed
        self.cmd_pub.publish(twist_msg)

if __name__ == '__main__':
    joy_teleop = JoyTeleop()
    rospy.spin()
