#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
class OdomSubscriber:
    def __init__(self):
        rospy.init_node('odom_subscriber', anonymous=True)
        self.odom_sub = rospy.Subscriber('/wamv/sensors/position/p3d_wamv', Odometry, self.odom_callback)
        self.vel_sub = rospy.Subscriber('/boat/odom', Odometry, self.vel_callback)
        self.contro_sub = rospy.Subscriber('/boat/cmd_vel', Twist, self.control_callback)
        self.pose_sub = rospy.Subscriber("/boat/pose_d", PoseStamped, self.position_callback)

        self.max_length = 700  # Longitud máxima de datos en el eje x

        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.vels_u = []
        self.vels_w = []
        self.controls_u = []
        self.controls_w = []
        self.times = []
        self.Xs_d = []
        self.Ys_d = []

        self.x_position = 0
        self.y_position = 0
        self.z_position = 0
        self.vel_u = 0
        self.vel_w = 0
        self.control_u = 0
        self.control_w = 0
        self.time = 0
        self.X_d = 0
        self.Y_d = 0
        self.error=0

        # Crear la figura y los subplots
        self.fig, self.axs = plt.subplots(2, 2)
        # Establecer los límites de los ejes
        
    

    def update_data(self, event):
        if len(self.times) > self.max_length:
            del self.x_positions[0]
            del self.y_positions[0]
            del self.z_positions[0]
            del self.vels_u[0]
            del self.vels_w[0]
            del self.controls_u[0]
            del self.controls_w[0]
            del self.times[0]
            del self.Xs_d[0]
            del self.Ys_d[0]
        self.time += 100
        self.x_positions.append(self.x_position)
        self.y_positions.append(self.y_position)
        self.z_positions.append(self.z_position)
        self.vels_u.append(self.vel_u)
        self.vels_w.append(self.vel_w)
        self.controls_u.append(self.control_u)
        self.controls_w.append(self.control_w)
        self.Xs_d.append(self.X_d)
        self.Ys_d.append(self.Y_d)
        self.times.append(self.time)
        self.error=str(math.sqrt((self.y_position-self.Y_d)**2+(self.x_position-self.X_d)**2))


        # Actualizar gráficos en tiempo real
        self.axs[0, 0].clear()
        self.axs[0, 0].plot(self.times, self.x_positions, 'b', label='Xr')
        self.axs[0, 0].plot(self.times, self.Xs_d, 'r', label='Xd')
        self.axs[0, 0].set_title('X position')
        self.axs[0, 0].set_ylabel('meters')
        self.axs[0, 0].legend()

        self.axs[0, 1].clear()
        self.axs[0, 1].plot(self.times, self.y_positions, 'b', label='Yr')
        self.axs[0, 1].plot(self.times, self.Ys_d, 'r', label='Yd')
        self.axs[0, 1].set_title('Y position')
        self.axs[0, 1].set_ylabel('meters')
        self.axs[0, 1].legend()

        self.axs[1, 0].clear()
        self.axs[1, 0].plot(self.times, self.vels_u, 'b', label='Ur')
        self.axs[1, 0].plot(self.times, self.controls_u, 'r', label='Uc')
        self.axs[1, 0].set_title('U Velocities')
        self.axs[1, 0].set_ylabel('m/s')
        self.axs[1, 0].legend()

        self.axs[1, 1].clear()
        self.axs[1, 1].plot(self.times, self.vels_w, 'b', label='Wr')
        self.axs[1, 1].plot(self.times, self.controls_w, 'r', label='Wc')
        self.axs[1, 1].set_title('W Velocities')
        self.axs[1, 1].set_ylabel('rad/s')
        self.axs[1, 1].legend()

        self.axs[0, 0].set_ylim(-50, 50)
        self.axs[0, 1].set_ylim(-50, 50)
        self.axs[1, 0].set_ylim(-2.2, 2.2)
        self.axs[1, 1].set_ylim(-0.8, 0.8)
        #print("Error: "+self.error)

    def odom_callback(self, msg):
        self.x_position = msg.pose.pose.position.x
        self.y_position = msg.pose.pose.position.y
        self.z_position = msg.pose.pose.position.z
    
    def vel_callback(self, msg):
        self.vel_u = msg.twist.twist.linear.x
        self.vel_w = msg.twist.twist.angular.z

    def control_callback(self, msg):
        self.control_u = msg.linear.x
        self.control_w = msg.angular.z
    def position_callback(self, data):
        self.X_d = data.pose.position.x
        self.Y_d= data.pose.position.y

if __name__ == '__main__':
    odom_subscriber = OdomSubscriber()

    # Actualizar datos en un hilo separado
    ani = FuncAnimation(odom_subscriber.fig, odom_subscriber.update_data, interval=10)
    plt.show()
