#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
import time
import sys

class ExcitationSignals:
    def __init__(self):
        # Inicialización del nodo
        rospy.init_node('excitation_signals', anonymous=True)
        self.flat=True
        # Publicador de las señales generales
        self.signals_pub = rospy.Publisher('/wamv/signals', Float32MultiArray, queue_size=10)
        self.stop_pub=rospy.Publisher("/stop_logging", Bool,  queue_size=10)  # Control de detención

        # Duración total de cada función (1 minuto por función, 20 funciones en total)
        self.duration_per_function = 10  # 60 segundos
        self.total_duration = 20 * self.duration_per_function  # 20 minutos

        # Frecuencia de publicación (cada 200 ms, o 5 Hz)
        self.publish_rate = rospy.Rate(10)  # 5 Hz

        # Lista de funciones de excitación
        self.excitation_functions = [
            self.excitation_function_1,   # Función 1: Seno simple
            self.excitation_function_2,   # Función 2: Suma de senos con diferentes frecuencias
            self.excitation_function_3,   # Función 3: Escalón alternado entre -1 y 1
            self.excitation_function_4,   # Función 4: Coseno con modulación
            self.excitation_function_5,   # Función 5: Señal aleatoria
            self.excitation_function_6,   # Función 6: Triangular con frecuencia variable
            self.excitation_function_7,   # Función 7: Seno con frecuencia creciente
            self.excitation_function_8,   # Función 8: Escalón aleatorio (amplitudes variables)
            self.excitation_function_9,   # Función 9: Pulsos alternados
            self.excitation_function_10,  # Función 10: Decaimiento exponencial
            self.excitation_function_11,  # Función 11: Diente de sierra con amplitud creciente
            self.excitation_function_12,  # Función 12: Cuadrada con periodo creciente
            self.excitation_function_13,  # Función 13: Seno modulado con frecuencia variable
            self.excitation_function_14,  # Función 14: Impulsos periódicos con detención
            self.excitation_function_15,  # Función 15: Coseno modulado con frecuencia descendente
            self.excitation_function_17,  # Función 17: Cuadrada con saltos abruptos
            self.excitation_function_18,  # Función 18: Pulsos crecientes
            self.excitation_function_19,  # Función 19: Seno decreciente
            self.excitation_function_20,   # Función 20: Impulsos alternados de larga duración
            self.excitation_function_21,   # Función 20: Impulsos alternados de larga duración
            self.excitation_function_22,  # Función 21: Termina y activa el guardado
            self.excitation_function_23, # Función 21: Termina y activa el guardado
            self.excitation_function_24,  # Función 21: Termina y activa el guardado
            self.excitation_function_25  # Función 21: Termina y activa el guardado
            
        ]

    def run(self):
        start_time = time.time()

        while not rospy.is_shutdown() and self.flat==True:
            elapsed_time = time.time() - start_time
            current_function_index = int((elapsed_time // self.duration_per_function) % len(self.excitation_functions))

            # Ejecución de la función de excitación actual
            left_cmd, right_cmd = self.excitation_functions[current_function_index](elapsed_time)
            
            # Publicar en el tópico '/wamv/signals'
            msg = Float32MultiArray()
            msg.data = [self.validar_y_saturar(left_cmd), self.validar_y_saturar(right_cmd)]  # Publicamos señales para los dos lados
            self.signals_pub.publish(msg)
            print(msg.data)
            # Mantener la frecuencia de publicación (5 Hz)
            self.publish_rate.sleep()
    def validar_y_saturar(self, numero):
        # Verifica si el número está en el rango -1 a 1
        if numero > 1:
            return 1
        elif numero < -1:
            return -1
        else:
            return numero
    # Definiciones de las funciones de excitación
    def excitation_function_1(self, t):
        """Seno simple con frecuencia base"""
        return np.sin(t), np.sin(t)

    def excitation_function_2(self, t):
        """Suma de senos con diferentes frecuencias"""
        return np.sin(t) + 0.5 * np.sin(2 * t), np.cos(t) + 0.5 * np.sin(3 * t)

    def excitation_function_3(self, t):
        """Escalón alternado entre -1 y 1 cada 5 segundos"""
        return (1 if (t // 5) % 2 == 0 else -1), (1 if (t // 5) % 2 == 1 else -1)

    def excitation_function_4(self, t):
        """Coseno con modulación de amplitud"""
        return np.cos(t) * (1 + 0.5 * np.sin(0.5 * t)), np.sin(t) * (1 + 0.5 * np.sin(0.5 * t))

    def excitation_function_5(self, t):
        """Señal aleatoria entre -1 y 1"""
        return np.random.uniform(-1, 1), np.random.uniform(-1, 1)

    def excitation_function_6(self, t):
        """Triangular con frecuencia variable"""
        return 2 * (t % 1) - 1, 2 * ((t + 0.5) % 1) - 1

    def excitation_function_7(self, t):
        """Seno con frecuencia creciente"""
        return np.sin(t * (1 + 0.1 * t)), np.cos(t * (1 + 0.1 * t))

    def excitation_function_8(self, t):
        """Escalón aleatorio con amplitudes entre -1 y 1"""
        return np.sign(np.sin(t)) * np.random.uniform(0.5, 1), np.sign(np.cos(t)) * np.random.uniform(0.5, 1)

    def excitation_function_9(self, t):
        """Pulsos alternados con periodos crecientes"""
        return (1 if t % 2 < 1 else -1), (1 if (t + 1) % 2 < 1 else -1)

    def excitation_function_10(self, t):
        """Decaimiento exponencial"""
        return np.exp(-0.1 * t) * np.sin(t), np.exp(-0.1 * t) * np.cos(t)

    def excitation_function_11(self, t):
        """Diente de sierra con amplitud creciente"""
        return 2 * (t % 2) - 1, 2 * ((t + 1) % 2) - 1

    def excitation_function_12(self, t):
        """Cuadrada con periodo creciente"""
        return np.sign(np.sin(t / (1 + 0.1 * t))), np.sign(np.cos(t / (1 + 0.1 * t)))

    def excitation_function_13(self, t):
        """Seno modulado con frecuencia variable"""
        return np.sin(t) * np.sin(0.5 * t), np.cos(t) * np.sin(0.5 * t)

    def excitation_function_14(self, t):
        """Impulsos periódicos con detención"""
        return (1 if (t // 10) % 2 == 0 else 0), (1 if (t // 10) % 2 == 1 else 0)

    def excitation_function_15(self, t):
        """Coseno con frecuencia descendente"""
        return np.cos(t / (1 + 0.1 * t)), np.sin(t / (1 + 0.1 * t))


    def excitation_function_17(self, t):
        """Cuadrada con saltos abruptos"""
        return np.sign(np.sin(t)) * np.random.uniform(0.5, 1), np.sign(np.cos(t)) * np.random.uniform(0.5, 1)

    def excitation_function_18(self, t):
        """Pulsos crecientes"""
        return (1 if t % 1 < 0.5 else -1) * (1 + 0.1 * t), (1 if (t + 0.5) % 1 < 0.5 else -1) * (1 + 0.1 * t)

    def excitation_function_19(self, t):
        """Seno decreciente"""
        return np.sin(t) / (1 + 0.1 * t), np.cos(t) / (1 + 0.1 * t)

    def excitation_function_20(self, t):
        """Impulsos alternados de larga duración"""
        return (1 if (t // 20) % 2 == 0 else 0), (1 if (t // 20) % 2 == 1 else 0)
    def excitation_function_21(self, t):
        """Impulsos alternados de larga duración"""
        return (-1 if (t // 20) % 2 == 0 else 0), (-1 if (t // 20) % 2 == 1 else 0)
    def excitation_function_22(self, t):
        """Impulsos alternados de larga duración"""
        return (1), (1)
    def excitation_function_23(self, t):
        """Impulsos alternados de larga duración"""
        return (-1), (-1)
    def excitation_function_24(self, t):
        return (0), (0)
    def excitation_function_25(self, t):
        print("pausa")
        stop_msg = Bool()
        stop_msg.data = True
        
        # Publicar el mensaje
        self.stop_pub.publish(stop_msg)
        self.flat==False
        sys.exit()



if __name__ == '__main__':
    try:
        controller = ExcitationSignals()
        controller.run()
    except rospy.ROSInterruptException:
        pass
