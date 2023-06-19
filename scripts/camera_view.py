import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convertir el mensaje de ROS a una imagen de OpenCV
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(e)
    else:
        # Mostrar la imagen en una ventana
        cv2.imshow("Cámara de ROS", cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('camera_viewer')
    
    # Suscribirse al tópico de la cámara
    rospy.Subscriber('/wamv/sensors/cameras/front_camera/image_raw', Image, image_callback)
    
    # Iniciar el bucle principal de ROS
    rospy.spin()

    # Cerrar la ventana al finalizar
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
