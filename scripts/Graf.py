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
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update_data)

        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.vels_u = []
        self.vels_w = []
        self.controls_u = []
        self.controls_w = []
        self.times = []
        self.Xs_d=[]
        self.Ys_d=[]

        self.x_position = 0
        self.y_position = 0
        self.z_position = 0
        self.vel_u = 0
        self.vel_w = 0
        self.control_u = 0
        self.control_w = 0
        self.time = 0
        self.X_d=100
        self.Y_d=100

    def update_data(self, event):

        self.time += 100
        self.x_positions.append(self.x_position)
        self.y_positions.append(self.y_position)
        self.z_positions.append(self.z_position)
        self.vels_u.append(self.vel_u)
        self.vels_w.append(self.vel_w)
        self.controls_u.append(self.control_u)
        self.controls_w.append(self.controls_w)
        self.Xs_d.append(self.X_d)
        self.Ys_d.append(self.Y_d)
        self.times.append(self.time)
        self.update_plot()





    def odom_callback(self, msg):
        self.x_position=msg.pose.pose.position.x
        self.y_position=msg.pose.pose.position.y
        self.z_position=msg.pose.pose.position.z
    
    def vel_callback(self, msg):

        self.vel_u=msg.twist.twist.linear.x
        self.vel_w=msg.twist.twist.angular.z

    def control_callback(self, msg):
        self.control_u=msg.twist.twist.linear.x
        self.control_w=msg.twist.twist.angular.z
        


    def update_plot(self):

                # Configurar la figura y los ejes
        fig, axs = plt.subplots(2, 2)
        while not rospy.is_shutdown():
        # Limpiar las gráficas
            for ax in axs.flatten():
                ax.clear()
            # Gráfica 1: X position
            axs[0, 0].plot(self.times, self.x_positions, 'b', label='Xr')
            axs[0, 0].plot(self.times, self.Xs_d, 'r', label='Xd')
            axs[0, 0].set_title('X position')
            axs[0, 0].set_ylabel('X position')
            axs[0, 0].set_xlabel('ms')
            axs[0, 0].legend()

            # Gráfica 2: Y position
            axs[0, 1].plot(self.times, self.y_positions, 'b', label='Yr')
            axs[0, 1].plot(self.times, self.Ys_d, 'r', label='Yd')
            axs[0, 1].set_title('Y position')
            axs[0, 1].set_ylabel('meters')
            axs[0, 1].set_xlabel('ms')
            axs[0, 1].legend()

            # Gráfica 3: U Velocities
            axs[1, 0].plot(self.times, self.vels_u, 'b', label='Ur')
            axs[1, 0].plot(self.times, self.controls_u, 'r', label='Uc')
            axs[1, 0].set_title('U Velocities')
            axs[1, 0].set_ylabel('m/s')
            axs[1, 0].set_xlabel('ms')
            axs[1, 0].legend()

            # Gráfica 4: W Velocities
            axs[1, 1].plot(self.times, self.vels_w, 'b', label='Wr')
            axs[1, 1].plot(self.times, self.controls_w, 'r', label='Wc')
            axs[1, 1].set_title('W Velocities')
            axs[1, 1].set_ylabel('rad/s')
            axs[1, 1].set_xlabel('ms')
            axs[1, 1].legend()

            # Ajustar el diseño
            plt.tight_layout()

            # Mostrar la gráfica
            plt.show()
            plt.pause(0.01)



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    odom_subscriber = OdomSubscriber()
    odom_subscriber.run()
