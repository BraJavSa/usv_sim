#!/usr/bin/env python3
import rospy
from std_msgs.msg import Empty

def publish_data():
    # Crear el nodo ROS
    rospy.init_node('sample_publisher', anonymous=True)

    # Crear el publicador para el tópico /sample
    publisher = rospy.Publisher('/sample', Empty, queue_size=10)

    # Establecer la tasa de publicación a 50 Hz
    rate = rospy.Rate(50)  # 50 Hz

    while not rospy.is_shutdown():
        msg = Empty()
        publisher.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_data()
    except rospy.ROSInterruptException:
        pass
