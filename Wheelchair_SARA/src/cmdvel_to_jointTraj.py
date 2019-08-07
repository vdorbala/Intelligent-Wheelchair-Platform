#!/usr/bin/env python


from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist, Pose
import rospy , time
import numpy as np
import time

def translate(value, leftMin, leftMax, rightMin, rightMax):
   # Figure out how 'wide' each range is
   leftSpan = leftMax - leftMin
   rightSpan = rightMax - rightMin

   # Convert the left range into a 0-1 range (float)
   valueScaled = float(value - leftMin) / float(leftSpan)

   # Convert the 0-1 range into a value in the right range.
   return rightMin + (valueScaled * rightSpan)

def callback(data):

    speed = translate(data.linear.x,-1,1,-1,1)

    SPEED = 'md: {}\r\n'.format(speed)

    angle = translate(data.angular.z,-1,1,1,-1)

    ANGLE = 'mt: {}\r\n'.format(angle)

    pub = rospy.Publisher("joint_trajectory", JointTrajectory)

    r = rospy.Rate(10)

    speed1 = 0
    speed2 = 0

    joint = JointTrajectory()

    jointpoints = JointTrajectoryPoint()

    jointpoints.velocities

    joint.joint_names.append('1')
    jointpoints.velocities.append(speed)
    joint.joint_names.append('2')
    jointpoints.velocities.append(speed)
    joint.points.append(jointpoints)

    # pub.publish(joint)
   
    if angle > 0:
        speed1 = -abs(angle)
        speed2 = abs(angle)

    if angle < 0:
        speed1 = abs(angle)
        speed2 = -abs(angle)

    if(angle > 0):
        print "right"
        del jointpoints.velocities[:]
        jointpoints.velocities.append(speed + speed1)
        jointpoints.velocities.append(speed + speed2)
        joint.points.append(jointpoints)
        # pub.publish(joint)

    elif (angle < 0):
        print "left"
        del jointpoints.velocities[:]
        jointpoints.velocities.append(speed + speed1)
        jointpoints.velocities.append(speed + speed2)
        joint.points.append(jointpoints)
        # pub.publish(joint)


    #saber.text('m2:startup')

    print (jointpoints)
    #MD: 0\r\n
    print SPEED
    print ANGLE

    # while not rospy.is_shutdown():
    pub.publish(joint)
    time.sleep(0.5)
        # r.sleep()


    #saber.text(SPEED)

    pass
    #print message


def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("cmd_vel", Twist, callback)

    rospy.spin()

if __name__ == '__main__':
    
    listener()

