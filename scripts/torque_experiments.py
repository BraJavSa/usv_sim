import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
import scipy.io as sio
import time
from geometry_msgs.msg import Vector3
import os

class TorqueExperiment:
    def __init__(self):
        # Inicialización de ROS
        rospy.init_node('torque_experiment', anonymous=True)

        # Configuración de tópicos de thrust y odometría
        self.cmd_pub = rospy.Publisher('/boat/cmd_thruster', Vector3, queue_size=10)

        # Variables para almacenar datos
        self.linear_velocity_data = []
        self.angular_velocity_data = []
        self.left_data = []
        self.right_data = []
        self.thrust_levels = np.linspace(0, 1, 10)  # Niveles de thrust de 0 a 1
        self.duration = 5  # Duración en segundos para cada nivel de thrust

        # Suscripción al tópico de odometría
        self.odom_sub = rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)
        self.left=0
        self.right=0

        

    def odom_callback(self, msg):
        """Callback para guardar datos de velocidad"""
        self.linear_velocity_data.append(msg.twist.twist.linear.x)
        self.angular_velocity_data.append(msg.twist.twist.angular.z)
        self.left_data.append(self.left)
        self.right_data.append(self.right)


    def apply_thrust(self, left_thrust, right_thrust):
        thrust_msg = Vector3()
        thrust_msg.x = left_thrust  # thrust izquierdo
        thrust_msg.y = right_thrust  # thrust derecho
        thrust_msg.z = 0.0  # Campo no utilizado
        self.cmd_pub.publish(thrust_msg)


    def run_experiment(self):
        """Realiza el experimento para medir velocidades lineales y angulares con diferentes niveles de thrust"""
        try:
            # Medición para velocidad lineal
            for thrust in self.thrust_levels:
                rospy.loginfo(f"Aplicando thrust lineal: {thrust}")
                start_time = rospy.Time.now().to_sec()
                
                # Aplicar thrust positivo en ambos lados
                while rospy.Time.now().to_sec() - start_time < self.duration:
                    self.apply_thrust(thrust, thrust)
                    self.left=thrust
                    self.right=thrust
                    time.sleep(0.1)  # Control de frecuencia
                
            # Restablecer thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)  # Pausa entre cada nivel de thrust
             # Medición para velocidad lineal
            for thrust in self.thrust_levels:
                rospy.loginfo(f"Aplicando thrust lineal: {thrust}")
                start_time = rospy.Time.now().to_sec()
                
                # Aplicar thrust positivo en ambos lados
                while rospy.Time.now().to_sec() - start_time < self.duration:
                    self.apply_thrust(-thrust, -thrust)
                    self.left=-thrust
                    self.right=-thrust
                    time.sleep(0.1)  # Control de frecuencia
                
            # Restablecer thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)  # Pausa entre cada nivel de thrust

            # Medición para velocidad angular (modo diferencial)
            for thrust in self.thrust_levels:
                rospy.loginfo(f"Aplicando thrust diferencial: {thrust}")
                start_time = rospy.Time.now().to_sec()
                
                # Aplicar thrust diferencial (positivo en un lado, negativo en el otro)
                while rospy.Time.now().to_sec() - start_time < self.duration:
                    self.apply_thrust(thrust, -thrust)
                    self.left=thrust
                    self.right=-thrust
                    time.sleep(0.1)
                
            # Restablecer thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)
            # Medición para velocidad angular (modo diferencial)
            for thrust in self.thrust_levels:
                rospy.loginfo(f"Aplicando thrust diferencial: {thrust}")
                start_time = rospy.Time.now().to_sec()
                
                # Aplicar thrust diferencial (positivo en un lado, negativo en el otro)
                while rospy.Time.now().to_sec() - start_time < self.duration:
                    self.apply_thrust(-thrust, thrust)
                    self.left=-thrust
                    self.right=thrust
                    time.sleep(0.1)
                
                # Restablecer thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)    

        finally:
            sio.savemat(os.path.expanduser('~/Documents/torque_velocity_data.mat'), {
                'linear_velocity_data': self.linear_velocity_data,
                'angular_velocity_data': self.angular_velocity_data,
                'left_data': self.left_data,
                'right_data': self.right_data
            })

            rospy.loginfo("Datos guardados en torque_velocity_data.mat")
            self.apply_thrust(0, 0)  # Asegurarse de que los thrusters estén en cero al finalizar

if __name__ == "__main__":
    experiment = TorqueExperiment()
    experiment.run_experiment()
