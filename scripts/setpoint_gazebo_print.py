#!/usr/bin/env python3

import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion

def load_gazebo_model(model_pose_x, model_pose_y):
    model_name = 'win_point'
    model_file = 'model.sdf'
    # Inicializa el nodo de ROS (si aún no está inicializado)

    # Espera a que el servicio "/gazebo/spawn_sdf_model" esté disponible
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    
    try:
        # Crea un objeto del servicio para cargar el modelo
        spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # Lee el contenido del archivo del modelo
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('vrx_gazebo')
        model_path = package_path +"/models/"+model_name+"/" + model_file
        with open(model_path, 'r') as file:
            model_xml = file.read()
        
        # Define la posición y orientación del modelo en el mundo
        model_pose_msg = Pose(
            position=Point(x=model_pose_x, y=model_pose_y, z=0.01),
            orientation=Quaternion(x=0, y=0, z=0, w=1)
        )
        
        # Llama al servicio para cargar el modelo
        response = spawn_model_service(model_name, model_xml, "", model_pose_msg, "world")
        
        # Verifica la respuesta para asegurarse de que el modelo se cargó correctamente
        if response.success:
            print(f"Modelo '{model_name}' cargado correctamente.")
        else:
            print(f"No se pudo cargar el modelo '{model_name}'.")
    except rospy.ServiceException as e:
        print(f"Error al llamar al servicio: {e}")

if __name__ == '__main__':
    # Reemplaza 'pos_d', 'model.sdf' y los valores de posición y orientación con los adecuados

    model_pose_x = 0 # Posición (x, y, z) y Orientación (x, y, z, w) del modelo
    model_pose_y = 0  # Posición (x, y, z) y Orientación (x, y, z, w) del modelo

    load_gazebo_model(model_pose_x, model_pose_y)
