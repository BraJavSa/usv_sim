#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped


class UpdateNode:
    def __init__(self):
        rospy.init_node("update_pose_node")
        self.pose_pub= rospy.Publisher("boat/pose_d", PoseStamped, queue_size=10)     
        self.pose_sub = rospy.Subscriber("boat/posed", PoseStamped, self.update_position)
        self.var = PoseStamped()
 
 
    def update_position(self, var):
    	self.var = var

    def send_initial_velocity(self):
        
        self.var.pose.position.x=0
        self.var.pose.position.y=0
        self.var.pose.position.z=0
        self.var.pose.orientation.x=0
        self.var.pose.orientation.y=0
        self.var.pose.orientation.z=0
        self.var.pose.orientation.w=1
        self.pose_pub.publish(self.var)



    def process(self):   
        self.pose_pub.publish(self.var)

def main():
    manual_mode = UpdateNode()
    manual_mode.send_initial_velocity()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        manual_mode.process()
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except IOError as e:
        print(e)