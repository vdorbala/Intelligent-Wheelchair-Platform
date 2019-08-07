#!/usr/bin/env python

from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
import rospy , time
import serial.tools.list_ports as port
import os
import time
from pysabertooth import Sabertooth

print "\nDetecting sabertooth....\n"
pl = list(port.comports())
print pl
address = ''
for p in pl:
  print p
  if 'Sabertooth' in str(p):
      address = str(p).split(" ")
print "\nAddress found @"
print address[0]
speed1 = 0
speed2 = 0

saber = Sabertooth(address[0], baudrate=9600, address=128, timeout=0.1)


def translate(value, leftMin, leftMax, rightMin, rightMax):
   # Figure out how 'wide' each range is
   leftSpan = leftMax - leftMin
   rightSpan = rightMax - rightMin

   # Convert the left range into a 0-1 range (float)
   valueScaled = float(value - leftMin) / float(leftSpan)

   # Convert the 0-1 range into a value in the right range.
   return rightMin + (valueScaled * rightSpan)

def callback(data):
  
    speed = translate(data.linear.x,-1,1,-60,60)
    #SPEED = 'md: {}\r\n'.format(speed) 
    angle = translate(data.angular.z,-1,1,40,-40)
    #ANGLE = 'mt: {}\r\n'.format(angle)


    speed_left = 0
    speed_right = 0
    saber.drive(1,speed)
    saber.drive(2,speed)
   
    if angle > 0:
        speed_left = -abs(angle)
        speed_right = abs(angle)

    if angle < 0:
        speed_left = abs(angle)
        speed_right = -abs(angle)

    if(angle > 0):
        print "right"
        saber.drive(1, speed + speed_left)
        saber.drive(2, speed + speed_right)
    elif (angle < 0):
        print "left"
        saber.drive(1, speed + speed_left)
        saber.drive(2, speed + speed_right)

    pass



def listener():

    #travel();
    rospy.init_node('base_controller', anonymous=True)
    rospy.Subscriber("cmd_vel", Twist, callback)
    rospy.spin()

if __name__ == '__main__':
    
    listener()

