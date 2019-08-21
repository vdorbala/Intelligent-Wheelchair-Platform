# File for programming Sabertooth commands on the wheelchair.

#!/usr/bin/env python
#import arduinoserial
from pysabertooth import Sabertooth
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
import rospy , time
import serial.tools.list_ports as port
#import pylcdlib
import pyttsx
from gtts import gTTS
import os
import speech_recognition as sr
import time
import subprocess
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

def forward():
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 1.29687123005
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
	print "forward"
	saber.drive(1,12)
	saber.drive(2,12)
	time.sleep(2);
	saber.drive(1,0)
	saber.drive(2,0)
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)

def backward():
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: -1.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
#     time.sleep(2);
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
	print "backward"
	saber.drive(1,-12)
	saber.drive(2,-12)
	time.sleep(2);
	saber.drive(1,0)
	saber.drive(2,0)

def left():
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 1.26576775642" 
# """)
#     time.sleep(2);
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
	print "left"
	saber.drive(1,-15)
	saber.drive(2,15)
	time.sleep(2);
	saber.drive(1,0)
	saber.drive(2,0)

def right():
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: -1.26576775642" 
# """)
#     time.sleep(2);
#     os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
	print "right"
	saber.drive(1,15)
	saber.drive(2,-15)
	time.sleep(2);
	saber.drive(1,0)
	saber.drive(2,0)


def stop():
	tts = gTTS(text='Okay. I am stopping now.', lang='en')
	tts.save("good.mp3")
	os.system("mpg321 good.mp3")
	os.system("""rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear:
#   x: 0.0
#   y: 0.0
#   z: 0.0
# angular:
#   x: 0.0
#   y: 0.0
#   z: 0.0" 
# """)
	print "stop"
	saber.drive(1,0)
	saber.drive(2,0)
	# time.sleep(2);
	# saber.drive(1,0)
	# saber.drive(2,0)

# engine = pyttsx.init()
# engine.say('Hello there. Good morning!')
# engine.runAndWait()

# tts = gTTS(text='Hello there. Good morning!', lang='en')
# tts.save("good.mp3")
# os.system("mpg321 good.mp3")

print "\n Initializing Kinect ... \n"

#command="roslaunch kinect2_bridge kinect2_bridge.launch"
#os.system("gnome-terminal -e 'bash -c \"roslaunch kinect2_bridge kinect2_bridge.launch; exec bash\"'")

print "\n Initialization complete"

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
# Callback function whenever a /cmd_vel is received from RoS
    #print data
    saber.text('m1:startup')
    saber.text('1,p100')
    saber.text('md:1000\r\n')

    saber.text('m2:startup')
    saber.text('r1:1940\r\n')
    saber.text('r2:1940\r\n')

    speed = translate(data.linear.x,-1,1,-60,60)
    #speed = translate(data.linear.x,-1,1,-2047,2047)
    SPEED = 'md: {}\r\n'.format(speed)
    angle = translate(data.angular.z,-1,1,20,-20)
    #angle = str(translate(-data.angular.z,-1,1,2047,-2047))
    ANGLE = 'mt: {}\r\n'.format(angle)
    #saber.text('m1:startup')
    #saber.drive(1,speed)
    #saber.drive(2,speed)
    if angle+speed > 20:
        speed1 = 20
    else :
        speed1 = angle+ speed
    if speed-angle < -20:
        speed2 = -20
    else :
        speed2 = speed-angle

	saber.drive(1,speed)
	saber.drive(2,speed)
   

    if(angle > 0):
        print "negative"
        saber.drive(1,speed2)
        saber.drive(2,speed1)
    elif (angle < 0):
        print "positive"
        saber.drive(1,speed2)
        saber.drive(2,speed1)

    #saber.text('m2:startup')
    #MD: 0\r\n
    print SPEED
    print ANGLE

    #saber.text(SPEED)

    pass
    #print message

def sabertoothStatusCallback(data):
    print data
    temperature = ('T [C]: {}'.format(saber.textGet('m2:gett')))

    saber.textGet('T,start')
    set_position = ('P : {}'.format(saber.textGet('T,p45')))

    battery = ('battery [mV]: {}'.format(saber.textGet('m2:getb')))
    print battery , temperature


def callback_kinect(data) :
    # pick a height
    height =  int (data.height / 2)
    # pick x coords near front and center
    middle_x = int (data.width / 2)
    # examine point
    middle = read_depth (middle_x, height, data)
    # do stuff with middle


def read_depth(width, height, data) :
    # read function
    if (height >= data.height) or (width >= data.width) :
        return -1
    data_out = pc2.read_points(data, field_names=None, skip_nans=False, uvs=[[width, height]])
    int_data = next(data_out)
    rospy.loginfo("int_data " + str(int_data))
    return int_data


def travel():

	while 1:	
		forward()
		stop()
		left()
		left()
		stop()
		forward()
		stop()
		left()
		left()
		stop()

# def callback2(data):
# 	threshold = 2048
# 	print "%d", data.data;
# 	if (data.data > threshold):
# 		saber.drive(1,0)
# 		saber.drive(2,0)
# 		tts = gTTS(text='There is an object in the way. I require assistance!', lang='en')
# 		tts.save("good.mp3")
# 		os.system("mpg321 good.mp3")



def listener():

	#travel();
	rospy.init_node('listener', anonymous=True)

    #rospy.Subscriber("/joy_teleop/cmd_vel", Twist, callback)
	rospy.Subscriber("cmd_vel", Twist, callback)
	#rospy.Subscriber("kinect2/sd/points", Image, callback2)
    #rospy.Subscriber("cmd_vel", Twist, callback)
 #    r = sr.Recognizer()
 #    with sr.Microphone() as source:
	# while 1:
	# 	audio = r.listen(source)
 #    	user = r.recognize_google(audio)
 #    	print(user)
 #    	if user in ("Stop","stop","Please Stop","please stop"):
 #    		stop();
    #sabertoothStatusCallback("as")
    #rospy.Subscriber("/mastercmd_vel", Twist, callback)
    # spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	
    listener()

