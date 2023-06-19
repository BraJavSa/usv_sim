import signal
import rospy

from xbox360controller import Xbox360Controller
from geometry_msgs.msg import Twist
rospy.init_node('keyboard_publisher')
pub = rospy.Publisher('boat/cmd_vel', Twist, queue_size=10)
twist_msg = Twist()
linear_vel_x = 0.0
linear_vel_y = 0.0
angular_vel = 0.0

def on_axis_moved(axis):
    print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))
    linear_vel_x = -axis.y
    angular_vel= axis.x*-3
    twist_msg.linear.x = linear_vel_x
    twist_msg.angular.z = angular_vel
    pub.publish(twist_msg)
try:
    with Xbox360Controller(0, axis_threshold=0.1) as controller:

        # Left and right axis move event
        controller.axis_l.when_moved = on_axis_moved
        #controller.axis_r.when_moved = on_axis_moved

        signal.pause()
except KeyboardInterrupt:
    pass
