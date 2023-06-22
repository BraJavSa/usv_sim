import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time

class PositionController:
    def __init__(self):
        self.hxd = 100
        self.hyd = 100
        self.uRef=0
        self.wRef=0
        self.hxe=0
        self.hye=0
        self.a=0.01 #punto de interes
        self.ts=0.01
        self.current_x=0
        self.current_y=0
        self.current_heading=0
        
        self.odom_subscriber = rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)
        self.vel_publisher = rospy.Publisher('/boat/cmd_vel', Twist, queue_size=10)

    def odom_callback(self, odom_msg):
        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.orientation = odom_msg.pose.pose.orientation.z

    def control(self):
        it=time.time()
        self.hxe=self.hxd-self.current_x
        self.hye=self.hyd-self.current_y
        
        he=np.array([[self.hxe],[self.hye]])
        J=np.array([[np.cos(self.current_heading),-self.a*np.sin(self.current_heading)],
                   [np.sin(self.current_heading), self.a*np.cos(self.current_heading)]])
        
        K=np.array([[0.1, 0],
                    [0, 0.1]])
        
        qpRef=np.linalg.pinv(J)@K@he

        uRef=qpRef[0][0]
        wRef=-qpRef[1][0]
        
        twist_msg = Twist()
        twist_msg.linear.x = uRef

        twist_msg.angular.z = wRef
        
        while (time.time()-it<self.ts):
            pass

        self.vel_publisher.publish(twist_msg)



# Inicialización de ROS
rospy.init_node('position_controller')

# Ejemplo de uso
controller = PositionController()

while not rospy.is_shutdown():
    controller.control()