#!/usr/bin/env python3
import rospy
import scipy.io as sio
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Bool
from std_msgs.msg import Empty  # Importar mensaje Empty
import time
import os
import sys


class DataLogger:
    def __init__(self):
        # Inicialización del nodo
        rospy.init_node('data_logger', anonymous=True)
        rospy.loginfo("Inicio de guardado")
        self.start_time = rospy.get_time()
        # Inicializar las variables de datos

        self.var_odom_lin_vel_x = 0.0
        self.var_odom_lin_vel_y = 0.0
        self.var_odom_ang_vel_z = 0.0


        self.var_left_thruster = 0.0
        self.var_right_thruster = 0.0
        self.prev_time = None  # Variable para el tiempo de la última llamada
        self.time = 0  # Variable para el tiempo entre callbacks en milisegundos
        self.data = {
            
            'odom_lin_vel_x': [],
            'odom_lin_vel_y': [],
            'odom_lin_vel_z': [],
            'odom_ang_vel_z': [],
            'left_thruster': [],
            'right_thruster': [],
        }

        # Suscriptores
        self.stop_logging = False  # Bandera para controlar el guardado

        rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/wamv/thrusters/left_thrust_cmd', Float32, self.left_thruster_callback)
        rospy.Subscriber('/wamv/thrusters/right_thrust_cmd', Float32, self.right_thruster_callback)
        rospy.Subscriber("/stop_logging", Bool, self.stop_logging_callback)  # Control de detención
        rospy.Subscriber("/sample", Empty, self.save_data_callback)  # Callback para guardar datos

        # Frecuencia de muestreo
        self.sample_rate = 50  # 100 Hz
        self.rate = rospy.Rate(self.sample_rate)

    def stop_logging_callback(self, msg):
        """Callback para manejar la bandera que detiene la recolección de datos"""
        self.stop_logging = msg.data
        if self.stop_logging:
            rospy.loginfo("Deteniendo la recolección de datos...")
            filepath = os.path.expanduser('~/Documents/usv_data.mat')
            # Guardar el diccionario de datos en formato .mat
            sio.savemat(filepath, self.data)
            rospy.loginfo(f'Datos guardados exitosamente en {filepath}')
            sys.exit()


    def odom_callback(self, msg):
        
        # Actualizar velocidades lineales y angulares
        self.var_odom_lin_vel_x = msg.twist.twist.linear.x
        self.var_odom_lin_vel_y = msg.twist.twist.linear.y
        self.var_odom_ang_vel_z = msg.twist.twist.angular.z



    def left_thruster_callback(self, msg):
        # Actualizar valor del propulsor izquierdo
        self.var_left_thruster = msg.data

    def right_thruster_callback(self, msg):
        # Actualizar valor del propulsor derecho
        self.var_right_thruster = msg.data

    def save_data_callback(self, msg):
        # Guardar datos de odometría
        current_time = rospy.get_time()  # Obtiene el tiempo actual

        # No hacer nada si no han pasado 15 segundos
        if current_time - self.start_time < 5:
            rospy.loginfo("Preparando...")
        else:
            
            self.data['odom_lin_vel_x'].append(self.var_odom_lin_vel_x)
            self.data['odom_lin_vel_y'].append(self.var_odom_lin_vel_y)
            self.data['odom_ang_vel_z'].append(self.var_odom_ang_vel_z)
            self.data['left_thruster'].append(self.var_left_thruster)
            self.data['right_thruster'].append(self.var_right_thruster)





    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    logger = DataLogger()
    logger.run()
