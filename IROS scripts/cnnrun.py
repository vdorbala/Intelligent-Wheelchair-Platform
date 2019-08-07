## Run with python 3
from __future__ import division
import os
from os import path
from multiprocessing import Process,Queue,JoinableQueue


import time
import sys
from pysabertooth import Sabertooth
import serial.tools.list_ports as port
import math

import numpy as np
from skimage import io, transform
from skimage.transform import rescale, resize

from cnn_finetune import make_model
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


import logging
from watchdog.observers import Observer
# from watchdog.events import LoggingEventHandler
from watchdog.events import FileSystemEventHandler

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,710)
fontScale              = 2
fontColor              = (0,270,0)
lineType               = 4

max_val = 3.9251497395*0.7897025648 - 0.0967433495
min_val = -4.538150963*0.7897025648 - 0.0967433495

warray = []
PATH1 =  "/home/vdorbala/ICRA/videos/Captured10/"

setGlobalLogger(None)

v_const = 0.2

pipe = 0

stop_name = "stop"
move_name = "move"
rev_name = "rev"


def deleteContent():
    with open(stop_name, "w"):
        pass
    with open(move_name, "w"):
        pass
    with open(rev_name, "w"):
        pass

class Event_Handler(FileSystemEventHandler):
    def __init__(self, arg):
        self.arg = arg
    def on_modified(self, event):
        # print('event type:' + event.event_type + 'path :' + event.src_path)
        f1 = open(stop_name,'r')
        f2 = open(move_name,'r')
        f3 = open(rev_name,'r')
        if event.src_path=="./stop":
            # f.truncate(0)
            fread1 = f1.read()
            f1.close()
            if fread1=="Stop":
                print("Stopping")
                callback_vel("STOP",self.arg)
                deleteContent()
        elif event.src_path=="./move":
            # f.truncate(0)
            fread2 = f2.read()
            f2.close()
            if fread2=="Move":
                print("Moving!")
                callback_vel("MOVE",self.arg)
                deleteContent()
        elif event.src_path=="./rev":
            # f.truncate(0)
            fread3 = f3.read()
            f3.close()
            if fread3=="Rev":
                print("Reversing")
                callback_vel("REV",self.arg)
                deleteContent()


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

    # print ("\nDetecting sabertooth....\n")
    # pl = list(port.comports())
    # print (pl)
    # address = ''
    # for p in pl:
    #   print (p)
    #   if 'Sabertooth' in str(p):
    #       address = str(p).split(" ")
    # print ("\nAddress found @")
    # print (address[0])

    # saber = Sabertooth(address[0], baudrate=9600, address=128, timeout=0.1)

    # print ("\n Initializing Kinect ...")

    # # command="roslaunch kinect2_bridge kinect2_bridge.launch"
    # # os.system("gnome-terminal -e 'bash -c \"roslaunch kinect2_bridge kinect2_bridge.launch; exec bash\"'")

    # imgtopic = "/kinect2/{}/image_color".format(quality)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu=1 
        print ("CUDA Available")
    else:
        device = torch.device("cpu")

    model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(227, 227))

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



def send_vel(w,saber,getval):
    global v_const, wl, wr

    speed = 15
    revspeed = -20
    # pipe = pipe_rec()
    # print ("INITIAL PIPE is " + str(pipe))
    print("GETVAL in FUNCTION IS "+str(getval))
    print("w received is " + str(w))

    w1 = w

    if w == 200:
        saber.drive(1,0)
        saber.drive(2,0)

    if w == 300:
        saber.drive(1,revspeed)
        saber.drive(2,revspeed)


    if w!=200 and w!=300:
        if w1>-0.01 and w <0:
            w = 0
        if w1<0.01 and w>0:
            w = 0

        # angle = translate(w,-3,3,-20,20)
        # print (angle)
        if w1<0:
            angle = -8
        else:
            angle = 8
        # print (speed)
        SPEED = 'md: {}\r\n'.format(speed)

        ANGLE = 'mt: {}\r\n'.format(angle)

        if getval==0:
            if (w > 0):
                print ("right")
                wr += 1
                # print (wr)
                saber.drive(1,speed)
                saber.drive(2,speed + 5 + 10*abs(w))
            elif (w < 0):
                print ("left")
                wl += 1
                saber.drive(1,speed + 5 + 10*abs(w))
                saber.drive(2,speed)
                
            elif(w == 0):
                print("straight!")
                saber.drive(1,speed) 
                saber.drive(2,speed)

    return


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

def main(q):
    global warray
    imgno =0
    run = 0
    num = 10
    # If RoS is being used
    # listener()
    model,saber,imgtopic = initialize()

    print ("q value is {}".format(q.get()))
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((227,227)),
    #     transforms.ToTensor(),
    # ])
    loopno = 0
    print (max_val,min_val)
    while loopno!=50:

        getval = q.get()

        # Just a DUMB Check

        # if getval!=0 or getval!=1 or getval!=2:
        #     print ("GETVAL is " + str(getval))
        #     continue
        

        loopno = loopno + 1
        # print (loopno)
        print ("GETVAL is " + str(getval))

        if getval == 0:

            image = get_image(imgtopic)
            image1 = image.copy()
            omega = vanishingpoint(image1)

            if image.shape[2]>3:
                image = image[:,:,:3]

            image = image_loader(image)

            # print ("Finished acquiring image")

            model.eval()

            with torch.no_grad():
                w = model(image)

            # print ("Before {}".format(w))

            w = ((w.cpu().numpy().flatten()))
            w = w*0.7897025648 - 0.0967433495
            # print ("w original is {}".format(w))
            # w = w #+ 0.3#+ 0.35655435
            # print ("w original is {}".format(w))
    # 
            # if (imgno%2==0):
            #     # cv2.putText(image1,'{}'.format(str(-1*w)[1:-1]),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            #     # cv2.imwrite(PATH1 + str(num) + ".png" ,image1)
            #     warray.append(w)
            #     num += 1
            # if getval==0:
            # send_vel(float(w/10),saber,getval)
            q.task_done()
            print ()

        elif getval==1:
            send_vel(200,saber,getval)
            q.task_done()
        elif getval==2:
            send_vel(300,saber,getval)           
            print("Waiting for obstacle to be cleared")
            send_vel(0,saber,getval)
            q.task_done()
    return imgno
# def pipe_set(pipe):
#     return pipe

# def pipe_rec():
#     global pipe
#     print ("Pipe value is " + str(pipe))
#     return pipe

def turn():

#Make a Uturn
    print("TURNING!")
    saber.drive(1,30)
    saber.drive(2,-30)
    time.sleep(2)
    saber.drive(1,0)
    saber.drive(2,0)
    time.sleep(1)



def callback_vel(command,q):
    if command == "STOP":
        pipe = 1

    elif command == "MOVE":
        pipe = 0

    elif command == "REV":
        pipe = 2

    q.put(pipe)
    # pipe1 = q.get()
    # print ("Pipe value is {}".format(pipe1))


def listener(q):
    print("Started")

    ## NO ROS FOR PYTHON3!!  :\

    # rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber("cmd_vel", Twist, callback_vel)
    # rospy.spin()
    pipe_path = "."
    # if not os.path.exists(pipe_path):
    #     os.mkfifo(pipe_path)
    # # Open the fifo. We need to open in non-blocking mode or it will stalls until
    # # someone opens it for writting
    # pipe_fd = open(pipe_path, "r")
    # # pipe_fd = open(pipe_path,"r")
    # # with os.fdopen(pipe_fd) as pipe:
    # while True:
    #     message = pipe_fd.read()
    #     print("MESSAGE IS "+ message)
    #     if message == "Stop":
    #         print("Object in the way! '%s'" % message)
    #         callback_vel(message)
    #     elif message == "Move":
    #         pipe=0
    #     time.sleep(2)

    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')
    # q.put(1)
    event_handler = Event_Handler(q)
    observer = Observer()
    observer.schedule(event_handler, path=pipe_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
            # print ("In Loop")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main_all(q):
    iterno=0
    print("Started")
    while(iterno!=20):
        main(q)
        # print ("Warray values are : \n Max = {} \n Min = {} \n Average = {}".format(max(warray),min(warray),sum(warray)/len(warray)))
        # file = open("/home/vdorbala/ICRA/videos/Test10","w+")
        # warray = [val for sublist in warray for val in sublist]
        # for no in range(len(warray)):   
        #     file.write("{},{}\n".format(no,warray[no]))
        turn()
        iterno = iterno+1
        print("Iteration number is "+ str(iterno))


if __name__ == '__main__':
    print ("Started")
    q = JoinableQueue(maxsize=1)
    # q.put(1)
    p1 = Process(target=main_all,args=(q,))
    p1.start()
    p2 = Process(target=listener,args=(q,))
    p2.start()
    p1.join()
    p2.join()
    q.join()