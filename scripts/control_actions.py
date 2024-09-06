#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

class ControlActions:
    def __init__(self):
        self.control_action = Twist()  # Acción de control inicial (velocidad lineal y angular)
        self.control_pub = rospy.Publisher('/usv/control_action', Twist, queue_size=10)
        rospy.Subscriber('/usv/control_input', Twist, self.control_callback)
        self.rate = rospy.Rate(10)  # 10 Hz = 100 ms

    def control_callback(self, msg):
        # Actualizar solo si se recibe un nuevo mensaje
        self.control_action = msg

    def run(self):
        # Inicializar con velocidades en cero
        self.control_action.linear.x = 0.0
        self.control_action.angular.z = 0.0

        while not rospy.is_shutdown():
            # Publicar la acción de control actual
            self.control_pub.publish(self.control_action)
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('control_actions', anonymous=True)
    control_actions = ControlActions()
    control_actions.run()

