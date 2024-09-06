#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class UpdatePosition:
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.Subscriber('/usv/pose', Pose, self.pose_callback)
        self.model_name = "wamv"

    def pose_callback(self, msg):
        model_state = ModelState()
        model_state.model_name = self.model_name
        model_state.pose = msg

        try:
            self.set_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('update_position', anonymous=True)
    update_position = UpdatePosition()
    rospy.spin()

