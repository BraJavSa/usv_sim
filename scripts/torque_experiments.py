import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import scipy.io as sio
import time

# Inicialización de ROS
rospy.init_node('torque_experiment', anonymous=True)

# Configuración de tópicos de thrust y odometría
left1_thruster_pub = rospy.Publisher('/wamv/thrusters/left1_thrust_cmd', Float64, queue_size=10)
left_thruster_pub = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float64, queue_size=10)
right1_thruster_pub = rospy.Publisher('/wamv/thrusters/right1_thrust_cmd', Float64, queue_size=10)
right_thruster_pub = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float64, queue_size=10)

# Variables para almacenar datos
linear_velocity_data = []
angular_velocity_data = []
thrust_levels = np.linspace(0, 1, 10)  # Niveles de thrust de 0 a 1
duration = 5  # Duración en segundos para cada nivel de thrust

# Callback para guardar datos de velocidad
def odom_callback(msg):
    linear_velocity_data.append(msg.twist.twist.linear.x)
    angular_velocity_data.append(msg.twist.twist.angular.z)

# Suscripción al tópico de odometría
odom_sub = rospy.Subscriber('/boat/odom', Odometry, odom_callback)

# Función para aplicar torque a los thrusters
def apply_thrust(left_thrust, right_thrust):
    left1_thruster_pub.publish(left_thrust)
    left_thruster_pub.publish(left_thrust)
    right1_thruster_pub.publish(right_thrust)
    right_thruster_pub.publish(right_thrust)

# Realiza el experimento
try:
    # Medición para velocidad lineal
    for thrust in thrust_levels:
        rospy.loginfo(f"Aplicando thrust lineal: {thrust}")
        start_time = rospy.Time.now().to_sec()
        
        # Aplicar thrust positivo en ambos lados
        while rospy.Time.now().to_sec() - start_time < duration:
            apply_thrust(thrust, thrust)
            time.sleep(0.1)  # Control de frecuencia
        
        # Restablecer thrusters a cero antes de cambiar el nivel
        apply_thrust(0, 0)
        time.sleep(1)  # Pausa entre cada nivel de thrust

    # Medición para velocidad angular (modo diferencial)
    for thrust in thrust_levels:
        rospy.loginfo(f"Aplicando thrust diferencial: {thrust}")
        start_time = rospy.Time.now().to_sec()
        
        # Aplicar thrust diferencial (positivo en un lado, negativo en el otro)
        while rospy.Time.now().to_sec() - start_time < duration:
            apply_thrust(thrust, -thrust)
            time.sleep(0.1)
        
        # Restablecer thrusters a cero antes de cambiar el nivel
        apply_thrust(0, 0)
        time.sleep(1)

finally:
    # Guardar los datos en un archivo .mat
    sio.savemat('torque_velocity_data.mat', {
        'linear_velocity_data': linear_velocity_data,
        'angular_velocity_data': angular_velocity_data,
        'thrust_levels': thrust_levels,
        'duration': duration
    })

    rospy.loginfo("Datos guardados en torque_velocity_data.mat")
    apply_thrust(0, 0)  # Asegurarse de que los thrusters estén en cero al finalizar
