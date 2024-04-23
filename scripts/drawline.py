import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

# Función para generar la trayectoria
def trajectory(t):
    Xd = 40 * np.sin(0.04 * t)
    Yd = 40 * np.sin(0.02 * t)
    return Xd, Yd

# Inicializar el nodo ROS
rospy.init_node('trajectory_publisher')

# Crear un objeto para publicar el marcador de la línea
marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)

# Esperar un segundo para asegurarse de que los servicios estén disponibles
rospy.sleep(1)

# Definir el intervalo de tiempo y la duración de la trayectoria
ts = 0.01  # Paso de tiempo
tf = 10    # Duración total

# Crear un mensaje de marcador para la línea
marker = Marker()
marker.header.frame_id = "world"
marker.header.stamp = rospy.Time.now()
marker.ns = "trajectory"
marker.id = 0
marker.type = Marker.LINE_STRIP
marker.action = Marker.ADD
marker.scale.x = 0.1  # Ancho de la línea

# Definir el color de la línea (en formato RGBA)
marker.color.r = 1.0
marker.color.g = 0.0
marker.color.b = 0.0
marker.color.a = 1.0  # Opacidad

# Definir la duración de la línea visual en el mundo (en segundos)
marker.lifetime = rospy.Duration()

# Generar puntos de la trayectoria y agregarlos al marcador de la línea
for t in np.arange(0, tf, ts):
    # Obtener las coordenadas de la trayectoria en el tiempo actual
    x, y = trajectory(t)

    # Agregar el punto actual a la línea visual
    point = Point()
    point.x = x
    point.y = y
    point.z = 0  # Altura constante, ajusta según sea necesario
    marker.points.append(point)

# Publicar el marcador de la línea
marker_pub.publish(marker)

# Mantener el nodo ROS activo
rospy.spin()
