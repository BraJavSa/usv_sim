import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import numpy as np

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.subscriber = rospy.Subscriber('/wamv/sensors/position/p3d_wamv', Odometry, self.odometry_callback)

    def odometry_callback(self, data):
        chasisPos_X = data.pose.pose.position.x
        chasisPos_Y = data.pose.pose.position.y
        quaterW = data.pose.pose.orientation.w
        quaterX = data.pose.pose.orientation.x 
        quaterY = data.pose.pose.orientation.y
        quaterZ = data.pose.pose.orientation.z 
        euler = self.quat2eulers(quaterW, quaterZ, quaterY, quaterX)
        orient = euler[0]
        signo = 1
        if orient < 0:
            signo = -1

        orient = -signo * (math.pi - signo * orient)
        chasisPos = [chasisPos_X, chasisPos_Y, orient]
        #print("Position")
        #print(chasisPos)

        chasisVel_W_X = data.twist.twist.linear.x 
        chasisVel_W_Y = data.twist.twist.linear.y
        chasisVel_W_w = data.twist.twist.angular.z
        vels = [chasisVel_W_X, chasisVel_W_Y, chasisVel_W_w]
        velsArr = np.array(vels)
        a=0.1
        JacChasis = [[math.cos(orient), -math.sin(orient), -a*math.sin(a+orient)],
                     [math.sin(orient), math.cos(orient), a*math.cos(a+orient)], [0, 0, 1]]
        invJac = np.linalg.inv(JacChasis)
        chasisVel = invJac.dot(velsArr)
        chasisVel[0] = -chasisVel[0]

        # Imprimir las velocidades calculadas
        #print("Velocity")
        print(chasisVel)

    def quat2eulers(self, q0, q1, q2, q3):

        roll = math.atan2(
        2 * ((q2 * q3) + (q0 * q1)),
        q0**2 - q1**2 - q2**2 + q3**2
        )  # radians
        pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
        yaw = math.atan2(
            2 * ((q1 * q2) + (q0 * q3)),
            q0**2 + q1**2 - q2**2 - q3**2
        )
        return (roll, pitch, yaw)

    def start(self):
        rospy.spin()

if __name__ == '__main__':
    controller = RobotController()
    controller.start()
