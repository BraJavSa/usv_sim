import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import os
import scipy.io as sio
import time

class TorqueExperiment:
    def __init__(self, duration=5, filename='torque_velocity_data.mat'):
        # Configuración de parámetros
        self.duration = duration
        self.filename = filename
        self.thrust_levels = np.linspace(0, 1, 10)  # Niveles de thrust de 0 a 1

        # Inicializa publicadores para los thrusters
        self.left1_thruster_pub = rospy.Publisher('/wamv/thrusters/left1_thrust_cmd', Float32, queue_size=10)
        self.left_thruster_pub = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.right1_thruster_pub = rospy.Publisher('/wamv/thrusters/right1_thrust_cmd', Float32, queue_size=10)
        self.right_thruster_pub = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size=10)
        self.left=0
        self.right=0
        # Variables para almacenar los datos
        self.linear_velocity_data = []
        self.angular_velocity_data = []
        self.left_data = []
        self.right_data = []
        self.signo=1
        # Subscripción al tópico de odometría
        self.odom_sub = rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)

    def odom_callback(self, msg):
        # Guarda las velocidades lineal y angular en las listas de datos
        self.linear_velocity_data.append(msg.twist.twist.linear.x)
        self.angular_velocity_data.append(msg.twist.twist.angular.z)
        self.left_data.append(self.left)
        self.right_data.append(self.right)

    def apply_thrust(self, left_thrust, right_thrust):
        # Publica el thrust en los thrusters izquierdo y derecho
        self.left1_thruster_pub.publish(left_thrust)
        self.left_thruster_pub.publish(left_thrust)
        self.right1_thruster_pub.publish(right_thrust)
        self.right_thruster_pub.publish(right_thrust)
        self.left=self.signo*left_thrust
        self.right=self.signo*right_thrust


    def run_experiment(self):
        # Ejecuta el experimento para velocidad lineal y luego para velocidad angular
        rospy.loginfo("Iniciando experimento de velocidad lineal")
        self.apply_linear_thrust()
        self.apply_linear2_thrust()

        rospy.loginfo("Iniciando experimento de velocidad angular (diferencial)")
        self.apply_angular_thrust()
        self.apply_angular2_thrust()


        # Guarda los datos en un archivo .mat
        self.save_data()
        rospy.loginfo(f"Datos guardados en {self.filename}")

    def apply_linear_thrust(self):
        # Aplica thrust simétrico para registrar la velocidad lineal
        for thrust in self.thrust_levels:
            rospy.loginfo(f"Aplicando thrust lineal: {thrust}")
            start_time = rospy.Time.now().to_sec()
            
            # Aplica thrust simétrico en ambos lados durante la duración especificada
            while rospy.Time.now().to_sec() - start_time < self.duration:
                self.apply_thrust(thrust, thrust)
                time.sleep(0.1)  # Control de frecuencia

            # Restablece los thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)  # Pausa entre cada nivel de thrust
    def apply_linear2_thrust(self):
        # Aplica thrust simétrico para registrar la velocidad lineal
        self.signo=-1
        for thrust in self.thrust_levels:
            rospy.loginfo(f"Aplicando thrust lineal: {thrust}")
            start_time = rospy.Time.now().to_sec()
            
            # Aplica thrust simétrico en ambos lados durante la duración especificada
            while rospy.Time.now().to_sec() - start_time < self.duration:
                self.apply_thrust(-thrust, -thrust)
                time.sleep(0.1)  # Control de frecuencia

            # Restablece los thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)  # Pausa entre cada nivel de thrust
    def apply_angular_thrust(self):
        # Aplica thrust diferencial para registrar la velocidad angular
        self.signo=1
        for thrust in self.thrust_levels:
            rospy.loginfo(f"Aplicando thrust diferencial: {thrust}")
            start_time = rospy.Time.now().to_sec()
            
            # Aplica thrust diferencial (positivo en un lado, negativo en el otro) durante la duración especificada
            while rospy.Time.now().to_sec() - start_time < self.duration:
                self.apply_thrust(thrust, -thrust)
                time.sleep(0.1)

            # Restablece los thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)
    def apply_angular2_thrust(self):
        # Aplica thrust diferencial para registrar la velocidad angular
        self.signo=-1
        for thrust in self.thrust_levels:
            rospy.loginfo(f"Aplicando thrust diferencial: {thrust}")
            start_time = rospy.Time.now().to_sec()
            
            # Aplica thrust diferencial (positivo en un lado, negativo en el otro) durante la duración especificada
            while rospy.Time.now().to_sec() - start_time < self.duration:
                self.apply_thrust(-thrust, thrust)
                time.sleep(0.1)

            # Restablece los thrusters a cero antes de cambiar el nivel
            self.apply_thrust(0, 0)
            time.sleep(1)
    def save_data(self):
        # Define la ruta y el nombre del archivo
        self.filename = os.path.expanduser('~/catkin_ws/src/usv_sim/MATLAB/torque_identification.mat')
        
        # Guarda los datos de velocidad en un archivo .mat
        sio.savemat(self.filename, {
            'linear_velocity_data': self.linear_velocity_data,
            'angular_velocity_data': self.angular_velocity_data,
            'left_cmd': self.left_data,
            'right_cmd': self.right_data
        })

if __name__ == '__main__':
    try:
        # Inicialización de ROS
        rospy.init_node('torque_experiment', anonymous=True)
        
        # Crea una instancia de TorqueExperiment y ejecuta el experimento
        experiment = TorqueExperiment()
        experiment.run_experiment()
        
        rospy.loginfo("Experimento completado")
        
    except rospy.ROSInterruptException:
        pass
    finally:
        # Asegura que los thrusters estén en cero al finalizar
        experiment.apply_thrust(0, 0)
