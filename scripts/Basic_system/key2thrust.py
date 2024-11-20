import rospy
from pynput import keyboard
from geometry_msgs.msg import Twist

# Inicializar ROS
rospy.init_node('keyboard_publisher')

# Publisher de ROS
pub = rospy.Publisher('boat/cmd_vel', Twist, queue_size=10)

# Velocidad del movimiento
linear_vel_x = 0.0
linear_vel_y = 0.0
angular_vel = 0.0

# Función para manejar el evento de presionar una tecla
def on_press(key):
    global linear_vel_x, linear_vel_y, angular_vel

    if key == keyboard.Key.w:
        linear_vel_x = 1.0
    elif key == keyboard.Key.s:
        linear_vel_x = -1.0
    elif key == keyboard.Key.a:
        angular_vel = 3.0
    elif key == keyboard.Key.d:
        angular_vel = -3.0

    publish_twist()

# Función para manejar el evento de soltar una tecla
def on_release(key):
    global linear_vel_x, linear_vel_y, angular_vel

    if key == keyboard.Key.w or key == keyboard.Key.s:
        linear_vel_x = 0.0
    elif key == keyboard.Key.a or key == keyboard.Key.d:
        angular_vel = 0.0

    publish_twist()

# Función para publicar el mensaje Twist en el topic "cmd_vel"
def publish_twist():
    twist_msg = Twist()
    twist_msg.linear.x = linear_vel_x
    twist_msg.linear.y = linear_vel_y
    twist_msg.angular.z = angular_vel
    pub.publish(twist_msg)

# Crear un listener de teclado
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Bucle principal de ROS
rospy.spin()