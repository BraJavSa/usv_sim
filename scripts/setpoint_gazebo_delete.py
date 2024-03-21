#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import DeleteModel

def delete_gazebo_model(model_name):
    # Inicializa el nodo de ROS (si aún no está inicializado)

    # Espera a que el servicio "/gazebo/delete_model" esté disponible
    rospy.wait_for_service('/gazebo/delete_model')
    
    try:
        # Crea un objeto del servicio para eliminar el modelo
        delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # Llama al servicio para eliminar el modelo
        response = delete_model_service(model_name)
        
        # Verifica la respuesta para asegurarse de que el modelo se eliminó correctamente
        if response.success:
            print(f"Modelo '{model_name}' eliminado correctamente.")
        else:
            print(f"No se pudo eliminar el modelo '{model_name}'.")
    except rospy.ServiceException as e:
        print(f"Error al llamar al servicio: {e}")

if __name__ == '__main__':
    # Reemplaza 'nombre_del_modelo' con el nombre del modelo que deseas eliminar
    delete_gazebo_model('win_point')
