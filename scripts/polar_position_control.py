import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class PositionController:
    def __init__(self):
        self.hxd = -50
        self.hyd = 50
        self.uRef=0
        self.wRef=0
        self.hxe=0
        self.hye=0
        self.a=0.01 #punto de interes
        self.current_x=0
        self.current_y=0
        self.orientation=0
        self.current_heading=0
        self.kv=1
        self.kk=0.4
        
        self.odom_subscriber = rospy.Subscriber('/boat/odom', Odometry, self.odom_callback)
        self.vel_publisher = rospy.Publisher('/boat/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)

    def odom_callback(self, odom_msg):
        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.orientation = odom_msg.pose.pose.orientation.z
    

    def control(self):
        self.hxe=self.hxd-self.current_x
        
        self.hye=self.hyd-self.current_y
        
        error=math.sqrt(self.hxe**2 + self.hye**2)
        a_e= math.atan2(self.hye, self.hxe)-self.orientation
        if error <=0.1:
            self.uRef=0
            self.wRef=0
        
        else:
            self.uRef=self.kv*math.tanh(error)*math.cos(a_e)
            self.wRef=self.kk*a_e+self.kv*(math.tanh(error)/error)*math.sin(a_e)*math.cos(a_e)

        twist_msg = Twist()
        twist_msg.linear.x = self.uRef
        twist_msg.angular.z = self.wRef
        self.vel_publisher.publish(twist_msg)
        self.rate.sleep()



# Inicialización de ROS
rospy.init_node('position_controller')

# Ejemplo de uso
controller = PositionController()

while not rospy.is_shutdown():
    controller.control()