#!/usr/bin/env python3

import rospy
from std_msgs.msg import Empty

class SamplePublisher:
    def __init__(self):
        # Inicialización del nodo
        rospy.init_node('sample_publisher', anonymous=True)

        # Crear un publicador para el tópico /sample
        self.pub = rospy.Publisher('/sample', Empty, queue_size=10)

        # Frecuencia de publicación (100 Hz)
        self.rate = rospy.Rate(50)

    def publish_sample(self):
        """Publica un mensaje vacío en el tópico /sample a 100 Hz"""
        empty_msg = Empty()  # Crear un mensaje Empty
        while not rospy.is_shutdown():
            # Publicar el mensaje en el tópico
            self.pub.publish(empty_msg)
            
            # Esperar para mantener la frecuencia de 100 Hz
            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Crear la instancia del publicador
        publisher = SamplePublisher()
        
        # Iniciar el bucle de publicación
        publisher.publish_sample()
    except rospy.ROSInterruptException:
        pass
