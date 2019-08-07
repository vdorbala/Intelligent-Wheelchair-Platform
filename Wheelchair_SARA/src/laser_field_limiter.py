#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan
import numpy as np
from geometry_msgs.msg import Twist
import os
import copy


def LaserScanProcess(data):

    scan_pub = rospy.Publisher('laser_scan', LaserScan, queue_size=50)
    scan = LaserScan()
    scan = copy.deepcopy(data)
    scan.ranges = data.ranges
    scan.intensities = data.intensities
    #print(scan.ranges[:180])
    a = scan.ranges[0:90]
    b = np.zeros(180)
    c = scan.ranges[270:359]

    scan.ranges = np.concatenate((a,b,c))
    # scan.ranges = a+b
    #print(len(a))
    r = rospy.Rate(5.0)
    scan_pub.publish(scan)
    r.sleep()



if __name__ == '__main__':
    try:
        rospy.init_node('laser_scan_publisher', anonymous=True)
        rospy.Subscriber("scan", sensor_msgs.msg.LaserScan, LaserScanProcess)
        # print("hi")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    