#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt
from threading import Thread

class OdomSubscriber:
    def __init__(self):
        rospy.init_node('odom_subscriber', anonymous=True)
        self.odom_sub = rospy.Subscriber('/wamv/sensors/position/p3d_wamv', Odometry, self.odom_callback)
        self.vel_sub = rospy.Subscriber('/boat/odom', Odometry, self.vel_callback)
        self.contro_sub = rospy.Subscriber('/boat/cmd', Twist, self.control_callback)
        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.vel_u = []
        self.vel_w = []
        self.control_u = []
        self.control_w = []
        self.times = []
        self.times1 = []
        self.init_time = 0
        self.init_time1 = 0
        self.barrera = False
        self.barrera1 = False

        # Valores diferentes de X_d para cada gráfica
        self.X_d_values = [100, 150, 75, 50]  # Ejemplo: 100 para x, 150 para x (repetido), 75 para y, 50 para z

    def odom_callback(self, msg):
        if not self.barrera:
            self.init_time = rospy.get_time()
            self.barrera = True
        current_time = rospy.get_time() - self.init_time
        self.times.append(current_time)
        self.x_positions.append(msg.pose.pose.position.x)
        self.y_positions.append(msg.pose.pose.position.y)
        self.z_positions.append(msg.pose.pose.position.z)
    
    def vel_callback(self, msg):
        if not self.barrera1:
            self.init_time1 = rospy.get_time()
            self.barrera1 = True
        current_time = rospy.get_time() - self.init_time1
        self.times1.append(current_time)
        self.vel_u.append(msg.twist.twist.linear.x)
        self.vel_w.append(msg.twist.twist.angular.z)

    def control_callback(self, msg):
        self.X_d_values[2]=msg.twist.twist.linear.x
        self.X_d_values[3]=msg.twist.twist.angular.z
        


    def update_plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle('Position vs. Time')

        # Configuración inicial de cada subgráfico
        axs_flat = axs.flat  # Aplanar el arreglo de subgráficos para iterar más fácilmente
        for i, ax in enumerate(axs_flat):
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            ax.set_ylim(0, 300)
            ax.axhline(y=self.X_d_values[i], color='r', linestyle='--', label=f'X_d = {self.X_d_values[i]}')
            ax.legend()

        plt.tight_layout()
        plt.ion()
        plt.show()

        while not rospy.is_shutdown():
            if not (self.barrera and self.barrera1):
                pass
            else:
                # Verificar si las listas tienen la misma longitud
                if len(self.times1) != len(self.vel_u) :
                    continue  # Salir de la iteración actual si las dimensiones son diferentes
                
                # Actualización de datos en cada subgráfico
                axs_flat[0].plot(self.times, self.x_positions, color='b', label='X Position')
                axs_flat[1].plot(self.times, self.y_positions, color='b', label='Y Position')  # Repetido para ejemplo
                axs_flat[2].plot(self.times1, self.vel_u, color='b', label='U velocity')
                axs_flat[3].plot(self.times1, self.vel_w, color='b', label='W velocity')
                axs_flat[2].set_ylim(-2.3, 2.3)
                axs_flat[3].set_ylim(-1, 1)
                plt.draw()
                plt.pause(0.01)



    def run(self):
        plot_thread = Thread(target=self.update_plot)
        plot_thread.start()
        rospy.spin()

if __name__ == '__main__':
    odom_subscriber = OdomSubscriber()
    odom_subscriber.run()
