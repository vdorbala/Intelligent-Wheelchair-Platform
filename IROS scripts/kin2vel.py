## Run with python 3
from __future__ import division
import os
from os import path
from multiprocessing import Process,Queue


import time
import sys
from pysabertooth import Sabertooth
import serial.tools.list_ports as port
import math

import numpy as np
from skimage import io, transform
from skimage.transform import rescale, resize

# from cnn_finetune import make_model
import cnn_finetune
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable


from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
from pylibfreenect2 import (Logger,
                            createConsoleLogger,
                            createConsoleLoggerWithDefaultLevel,
                            getGlobalLogger,
                            setGlobalLogger,
                            LoggerLevel)
# from pylsd.lsd import lsd

NUMBER=25
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import logging
from watchdog.observers import Observer
# from watchdog.events import LoggingEventHandler
from watchdog.events import FileSystemEventHandler


class MyHandler(FileSystemEventHandler):
    def __init__(self, arg):
        self.arg = arg
    def on_modified(self, event):
        # print('event type:' + event.event_type + 'path :' + event.src_path)
        if event.src_path=="./stop":
            print("Stopping")
            callback_vel("STOP",self.arg)
        elif event.src_path=="./move":
            print("Moving!")
            callback_vel("MOVE",self.arg)


setGlobalLogger(None)

v_const = 0.2
saber = 0

def get_image(imgtopic):

    image = imgtopic

    fn = Freenect2()
 
    num_devices = fn.enumerateDevices()
    assert num_devices > 0
 
    serial = fn.getDefaultDeviceSerialNumber()
    assert serial == fn.getDeviceSerialNumber(0)
 
    device = fn.openDevice(serial)
 
    assert fn.getDefaultDeviceSerialNumber() == device.getSerialNumber()
    device.getFirmwareVersion()
 
    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)
 
    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
 
    device.start()
 
    # Registration
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    # undistorted = Frame(512, 424, 4)
    # registered = Frame(512, 424, 4)
 
    # # optional parameters for registration
    # bigdepth = Frame(1920, 1082, 4)
    # color_depth_map = np.zeros((424, 512), np.int32)
 
    # # test if we can get two frames at least
    # frames = listener.waitForNewFrame()
    # listener.release(frames)
 
    # frames as a first argment also should work
    frames = FrameMap()
    listener.waitForNewFrame(frames)
 
    color = frames[FrameType.Color]
    # ir = frames[FrameType.Ir]
    # depth = frames[FrameType.Depth]
 
    # for frame in [ir, depth]:
    #     assert frame.exposure == 0
    #     assert frame.gain == 0
    #     assert frame.gamma == 0
 
    # for frame in [color]:
    #     assert frame.exposure > 0
    #     assert frame.gain > 0
    #     assert frame.gamma > 0
 
    # registration.apply(color, depth, undistorted, registered)
    # registration.apply(color, depth, undistorted, registered,
    #                    bigdepth=bigdepth,
    #                    color_depth_map=color_depth_map.ravel())
 
    # assert color.width == 1920
    # assert color.height == 1080
    # assert color.bytes_per_pixel == 4
 
    # assert ir.width == 512
    # assert ir.height == 424
    # assert ir.bytes_per_pixel == 4
 
    # assert depth.width == 512
    # assert depth.height == 424
    # assert depth.bytes_per_pixel == 4
 
    # assert color.asarray().shape == (color.height, color.width, 4)
    # assert ir.asarray().shape == (ir.height, ir.width)
    # assert depth.asarray(np.float32).shape == (depth.height, depth.width)
 
    return color.asarray()

def initialize():
    global saber
    PATH = 'direct2642_noisy_resnet2.pt' #direct2642_noisy_resnet2.pt, direct2642_noisy.pt
    quality = "qhd"

    print ("\n Initializing Kinect ...")

    # command="roslaunch kinect2_bridge kinect2_bridge.launch"
    # os.system("gnome-terminal -e 'bash -c \"roslaunch kinect2_bridge kinect2_bridge.launch; exec bash\"'")

    imgtopic = "/kinect2/{}/image_color".format(quality)

    print ("\nDetecting sabertooth....\n")
    pl = list(port.comports())
    print (pl)
    address = ''
    for p in pl:
      print (p)
      if 'Sabertooth' in str(p):
          address = str(p).split(" ")
    print ("\nAddress found @")
    print (address[0])

    saber = Sabertooth(address[0], baudrate=9600, address=128, timeout=0.1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu=1 
        print ("CUDA Available")
    else:
        device = torch.device("cpu")

    model = cnn_finetune.make_model('resnet18', num_classes=1, pretrained=True, input_size=(227, 227))

    if use_gpu == 1:
        model = nn.DataParallel(model).cuda()

    if path.exists(PATH):
        model.load_state_dict(torch.load(PATH))


    model = model.to(device)

    print ("\n Initialization complete")

    return model,saber,imgtopic

def translate(value, leftMin, leftMax, rightMin, rightMax):
   leftSpan = leftMax - leftMin
   rightSpan = rightMax - rightMin
   valueScaled = float(value - leftMin) / float(leftSpan)
   return rightMin + (valueScaled * rightSpan)

wl = 0
wr = 0

def send_vel(w,saber):
    global v_const, wl, wr

    w1 = w

    if w1>-0.01 and w <0:
        w = 0
    if w1<0.01 and w>0:
        w = 0

    # angle = translate(w,-3,3,-20,20)
    # print (angle)
    if w<0:
        angle = -8
    else:
        angle = 8

    # speed = translate(v_const,-1,1,-60,60)
    # if wr>2:
    #     w = -1*(w+8)
    #     wr = 0
    # if wl>2:
    #     w = -1*(w+8)
    #     wl = 0

    speed = 24
    # speed = 23 + 100*(abs(float(w)))

    # if speed>30:
    #     speed = 23

    # print("{}".format(speed))

    # speedright = speed*1.1

    # print (speed)
    SPEED = 'md: {}\r\n'.format(speed)

    ANGLE = 'mt: {}\r\n'.format(angle)

    # if angle+speed > 20:
    #     speed1 = 20
    # else:
    #     speed1 = angle + speed

    # if speed-angle < -20:
    #     speed2 = -20
    # else :
    #     speed2 = speed-angle

    # saber.drive(1,speed)
    # saber.drive(2,speed)

    # print (8*abs(w))
    # if w1!=0:
        # print ("Actual velocity is {}".format(angle))
    if (w > 0):
        print ("right")
        wr += 1
        # print (wr)
        saber.drive(1,speed)
        saber.drive(2,speed + 5 + 8*abs(w))
    elif (w < 0):
        print ("left")
        wl += 1
        saber.drive(1,speed + 5 + 8*abs(w))
        saber.drive(2,speed)
    elif(w == 0):
        print("straight!")
        saber.drive(1,speed) 
        saber.drive(2,speed)
## Using RoS for sabertooth control

            # def callback(data):
            #     #print data
            #     saber.text('m1:startup')
            #     saber.text('1,p100')
            #     saber.text('md:1000\r\n')

            #     saber.text('m2:startup')
            #     saber.text('r1:1940\r\n')
            #     saber.text('r2:1940\r\n')

            #     speed = translate(data.linear.x,-1,1,-60,60)
            #     #speed = translate(data.linear.x,-1,1,-2047,2047)
            #     SPEED = 'md: {}\r\n'.format(speed)
            #     angle = translate(data.angular.z,-1,1,20,-20)
            #     #angle = str(translate(-data.angular.z,-1,1,2047,-2047))
            #     ANGLE = 'mt: {}\r\n'.format(angle)
            #     #saber.text('m1:startup')
            #     #saber.drive(1,speed)
            #     #saber.drive(2,speed)
            #     if angle+speed > 20:
            #         speed1 = 20
            #     else :
            #         speed1 = angle+ speed
            #     if speed-angle < -20:
            #         speed2 = -20
            #     else :
            #         speed2 = speed-angle

            #     saber.drive(1,speed)
            #     saber.drive(2,speed)
               

            #     if(angle > 0):
            #         print "negative"
            #         saber.drive(1,speed2)
            #         saber.drive(2,speed1)
            #     elif (angle < 0):
            #         print "positive"
            #         saber.drive(1,speed2)
            #         saber.drive(2,speed1)

            #     #saber.text('m2:startup')
            #     #MD: 0\r\n
            #     print SPEED
            #     print ANGLE

            #     #saber.text(SPEED)

            #     pass

            # def listener():

            #     #travel();
            #     rospy.init_node('listener', anonymous=True)
            #     rospy.Subscriber("cmd_vel", Twist, callback)
            #     rospy.spin()



loader = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])

def image_loader(image):
    
    imsize = 224
    image = loader(image)
    image = torch.FloatTensor(image)
    # sample = {'image': image}
    image = image.unsqueeze(0) # For VGG
    image = Variable(image, requires_grad=True)
    return image.cuda()


def vanishingpoint(image):



    return image


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,710)
fontScale              = 2
fontColor              = (0,270,0)
lineType               = 4

max_val = 3.9251497395*0.7897025648 - 0.0967433495
min_val = -4.538150963*0.7897025648 - 0.0967433495

warray = []
PATH =  "/home/vdorbala/ICRA/videos/Captured10/"

def main():
    global warray,NUMBER
    imgno =0
    run = 0
    num = 10
    # If RoS is being used
    # listener()
    model,saber,imgtopic = initialize()

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((227,227)),
    #     transforms.ToTensor(),
    # ])

    print (max_val,min_val)

    for i in range(NUMBER):
        image = get_image(imgtopic)
        image1 = image.copy()
        omega = vanishingpoint(image1)

        if image.shape[2]>3:
            image = image[:,:,:3]
        # image =  cv2.resize(image,(227,227))

        # b_channel, g_channel, r_channel, a_channel = cv2.split(image)

        # image = cv2.merge((b_channel, g_channel, r_channel))
        # # image = torch.from_numpy(image)
        # image = cv2.flip (image,1)
        # image1 = cv2.flip(image1,1)
        # imgno += 1

        # image = np.reshape(image,(3,227,227))
        image = image_loader(image)

        # print ("Finished acquiring image")

        model.eval()

        with torch.no_grad():
            w = model(image)

        # print ("Before {}".format(w))

        w = ((w.cpu().numpy().flatten()))#*0.7897025648) - 0.0967433495
        w = w*0.7897025648 - 0.0967433495
        # print ("w original is {}".format(w))
        # w = w #+ 0.3#+ 0.35655435
        print ("w original is {}".format(w))

        if (imgno%2==0):
            # cv2.putText(image1,'{}'.format(str(-1*w)[1:-1]),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.imwrite(PATH + str(num) + ".png" ,image1)
            warray.append(w)
            num += 1
            send_vel(float(w/10),saber)
        # wnew =  [val for sublist in warray for val in sublist]
        if run == 0:
            run = 1
            continue

        # cv2.imwrite(PATH + str(imgno) + ".png" ,image1)
        # cv2.imshow('Window',image1)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()
        # time.sleep(0.5)
        # send_vel(0,saber)
        # time.sleep(2)
    return imgno


def turn():

#Make a Uturn
    saber.drive(1,30)
    saber.drive(2,-30)
    time.sleep(3)
    saber.drive(1,0)
    saber.drive(2,0)
    time.sleep(1)

def callback_vel(data):
    global saber
    send_vel(0,saber)


def listener():
    ## NO ROS FOR PYTHON3!!  :\
    # rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber("cmd_vel", Twist, callback_vel)
    # rospy.spin()
    pipe_path = "/tmp/pipe"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    # Open the fifo. We need to open in non-blocking mode or it will stalls until
    # someone opens it for writting
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Object in the way! '%s'" % message)
                callback_vel(message)
            # print("Doing other stuff")
            time.sleep(0.5)


def main_all():
    iterno=0
    while(iterno!=10):
        ino = main()
        print("Main ends")
        # print ("Warray values are : \n Max = {} \n Min = {} \n Average = {}".format(max(warray),min(warray),sum(warray)/len(warray)))
        # file = open("/home/vdorbala/ICRA/videos/Test10","w+")
        # warray = [val for sublist in warray for val in sublist]
        # for no in range(len(warray)):   
            # file.write("{},{}\n".format(no,warray[no]))
        turn()
        iterno = iterno+1
        print("Iteration number is "+str(iterno))

def capture():
    model,saber,imgtopic = initialize()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi',fourcc, 20.0,(1920,1080))
    # imgtopic = "/kinect2/{}/image_color".format(quality)
    num = 0
    while num!=100:
        image = get_image(imgtopic)
        if image.shape[2]>3:
            image = image[:,:,:3]
        out.write(image)
        num = num + 1
        time.sleep(0.033)
        # print ("\nWriting")
    out.release()
    # cap = cv2.VideoCapture(0)

    # while(cap.isOpened()):
    #     ret, frame = cap.read()

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     cv2.imshow('frame',gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':

    q = Queue()
    # p1 = Process(target=listener)
    # p1.start()
    # p2 = Process(target=main_all)
    # p2.start()
    p3 = Process(target=capture)
    p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    # q.join()