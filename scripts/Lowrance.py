import io
import pynmea2
import serial
import rospy
from std_msgs.msg import Float32

if __name__ == '__main__':
    rospy.init_node('random_float_publisher')
    pub = rospy.Publisher('/boat/dpt', Float32, queue_size=10)
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1.0)
    sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser))

    rate = rospy.Rate(1) # 1 Hz

    while not rospy.is_shutdown():
        try:
            line = sio.readline()
            msg = pynmea2.parse(line)
            aux=str(type(msg)).split(".")[3].split("'")[0]
            if aux=="DPT":
                profundidad=float(msg.depth)
                pub.publish(profundidad)
                rate.sleep()
        except serial.SerialException as e:
            print("---1")
            continue
        except pynmea2.ParseError as e:
            print("____2")
            continue

