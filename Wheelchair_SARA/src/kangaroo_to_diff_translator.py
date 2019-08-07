#!/usr/bin/env python

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int16
import rospy , time
import numpy as np


def callback(data):

    left_ticks = (data.position[0])*1440
    right_ticks = (data.position[1])*1440
    
    right_vel = data.velocity[0]
    left_vel = data.velocity[1]

    #print ("Ticks are :\n {}, {} \nVelocities are : \n {}, {}".format(left_ticks,right_ticks,left_vel,right_vel))

    lwheel = rospy.Publisher('lwheel', Int16, queue_size= 1)
    rwheel = rospy.Publisher('rwheel', Int16, queue_size = 1)

    lwheelval = Int16()
    rwheelval = Int16()

    del lwheelval.data
    del rwheelval.data

    # values.layout.data_offset = 50
    lwheelval.data = int(left_ticks)
    rwheelval.data = int(right_ticks)

    lwheel.publish(lwheelval)
    rwheel.publish(rwheelval)


def listener():

    rospy.init_node('listener')
    rospy.Subscriber("joint_state", JointState, callback)

    rospy.spin()

if __name__ == '__main__':
    
    listener()
