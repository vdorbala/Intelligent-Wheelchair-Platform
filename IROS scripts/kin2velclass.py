# Classical approach to corridor following
## Run with python 3
from __future__ import division
import os
from os import path

import time
import sys

sys.path.insert(0, '/Path/to/pylsd/')

from pysabertooth import Sabertooth
import serial.tools.list_ports as port
import math

import numpy as np
from skimage import io, transform
from skimage.transform import rescale, resize

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
from pylsd.lsd import lsd

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


# from cnn_finetune import make_model
# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms, utils
# from torch.autograd import Variable


from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
from pylibfreenect2 import (Logger,
                            createConsoleLogger,
                            createConsoleLoggerWithDefaultLevel,
                            getGlobalLogger,
                            setGlobalLogger,
                            LoggerLevel)
# from pylsd.lsd import lsd


ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

setGlobalLogger(None)

v_const = 0.2

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

    PATH = '/home/vdorbala/ICRA/direct2642_noisy_resnet2.pt' #direct2642_noisy_resnet2.pt, direct2642_noisy.pt
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

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     use_gpu=1 
    #     print ("CUDA Available")
    # else:
    #     device = torch.device("cpu")

    # model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(227, 227))

    # if use_gpu == 1:
    #     model = nn.DataParallel(model).cuda()

    # if path.exists(PATH):
    #     model.load_state_dict(torch.load(PATH))


    # model = model.to(device)
    model = 1

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

    speed = 18
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



# loader = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])

# def image_loader(image):
    
#     imsize = 224
#     image = loader(image)
#     image = torch.FloatTensor(image)
#     # sample = {'image': image}
#     image = image.unsqueeze(0) # For VGG
#     image = Variable(image, requires_grad=True)
#     return image.cuda()





## CLASSICAL METHOD


def thetam(pt1, pt2):
    t1,t2 = pt1[5],pt2[5]       # Lines corressponding to leftmost and rightmost.
    
    a1 = pt1[0]
    b1 = pt1[1]
    a2 = pt1[2]
    b2 = pt1[3]

    x1 = pt2[0]
    y1 = pt2[1]
    x2 = pt2[2]
    y2 = pt2[3]

    # t1 = abs(t1)
    # t2 = abs(t2)

    # t1 = math.pi - t1
    # t2 = math.pi - t2

    # print ("T1 is {}. T2 is {}".format(t1*180/math.pi,t2*180/math.pi))

    theta1 = abs(np.arctan((b1-b2)/(a1-a2)))
    theta2 = abs(np.arctan((y1-y2)/(x1-x2)))

    theta1 = math.pi/2 - theta1
    theta2 = -1*(math.pi/2 - theta2)

    # print "theta values are {0}, {1}".format(theta1*180/math.pi,theta2*180/math.pi)

    # tl = math.pi/2 - t1
    # tr = math.pi/2 - t2
    # if t1<0:
    #     t1 = t1 + math.pi
    # if t2<0:
    #     t2 = t2 + math.pi


    m1 = np.tan(theta1)
    m2 = np.tan(theta2)

    m = m1 + m2

    # print "L and R values are {0}, {1}".format(math.degrees(m1),math.degrees(m2))

    atan = m/2

    # print "atan is {0}".format(atan)
    tm = math.atan(atan)
    return tm

def angle(pt1, pt2):
    t1,t2 = pt1[5] , pt2[5]
    # if t1<0:
    #     t1 = t1 + math.pi
    # if t2<0:
    #     t2 = t2 + math.pi
    if t1<0:
        t1 = math.pi + t1
    if t2<0:
        t2 = math.pi + t2

    # print "The theta values are {0}, {1}".format(t1*180/math.pi,t2*180/math.pi)

    tb = abs(t1 - t2)
    ret = (min(t1,t2) + (tb/2))

    # print 'The return value is {0}'.format(ret)

    return ret

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def setvp(event,x,y,flags,param):
    global v_x,v_y,xfin,yfin,showagain

    showagain = 1

    if event == cv2.EVENT_LBUTTONDOWN:
        print ("Left is {},{}".format(x,y))

    elif event == cv2.EVENT_LBUTTONUP:
        v_x = x
        v_y = y

    if event == cv2.EVENT_RBUTTONDOWN:
        print ("Right is {},{}".format(x,y))

    elif event == cv2.EVENT_RBUTTONUP:
        xfin = x
        yfin = y


def process(lines,img,v_x,v_y):

    ii=1
    sel_lines = []
    thetas = []
    mags = []
    j = 0
    w = 2

    for i in xrange(lines.shape[0]):

        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))

        x1 = int(lines[i,0])
        x2 = int(lines[i,2])
        y1 = int(lines[i,1])
        y2 = int(lines[i,3])
        w =  int(lines[i, 4])

        # for pt1,pt2 in lines:
        #     if y1 < y_mean or y2 < y_mean:
        #         lines.remove(pt1)

        if (x1- x2)!=0: # Make sure lines are not collinear
            theta = np.arctan2((y2-y1),(x2-x1))

            m2= np.tan(theta)
            l_mag = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

            # Extend the lines to the entire image and compute the intersetion point
            c2 = y1 - m2*x1
            x3 = int(img.shape[1]/1.8 + x1) # 1000 was chosen arbitrarily (any number higher than half the image width) 
            y3 = int(m2*x3 + c2)
            x4 = int(x1 - img.shape[1]/1.8) # 1000 was chosen arbitrarily 
            y4 = int(m2*x4 + c2)


            # if y4<v_y:
            #     y4 = int(v_y)

            lines1 = lines[i]

            if abs(theta) > 0.1 and l_mag > 1 and abs(theta) < 1.4: # 1.5708 = 90 degrees . 0.2, 20, 1.1 for Test Sets 1,2,3,4 . 0.2, 30, 1.1 for MIT. 0.3, 40, 1.3 for CMU, 0.3,25,1.3 Umich_eq
            # CMU - 0.3, 20 ,1.4; MIT 0.3, 20, 1.3; Test 1-4 0.2,30,1.1; Umich_eq 0.3, 25, 1.3
                lines1 = np.append(lines1,theta)
                lines1 = np.append(lines1,l_mag)
                thetas = np.append(thetas,theta)
                mags = np.append(mags,l_mag)

                sel_lines.append(lines1)

                if y3>v_y or y4>v_y:
                    if y3>=y4:
                        y4 = v_y
                        x4 = int((y4 - c2)/m2)                         
                        # print 'y3 > y4: {0}, {1}, {2}, {3}'.format(x3,y3,x4,y4)   
                        pt11 = (x4,y4)
                        pt22 = (x3,y3)
                    else:
                        y3 = v_y
                        x3 = int((y3 - c2)/m2) 
                        # print 'y4 > y3: {0}, {1}, {2}, {3}'.format(x3,y3,x4,y4)
                        pt11 = (x3,y3)
                        pt22 = (x4,y4)

                sel_lines[j][0] = int(x3)
                sel_lines[j][1] = int(y3)
                sel_lines[j][2] = int(x4)
                sel_lines[j][3] = int(y4)

                j = j+1

    return sel_lines,thetas,mags,w


def classical(img):
    img = cv2.resize(img,(224,224)) # For Resnet18
    img1 = img.copy()
    img = cv2.flip (img,1)

    v_x = int(img.shape[1]/2)
    v_y = int(img.shape[0]/2)

    #Calculate lines using the LSD algorithm
    # v_x = int(v_x)
    # v_y = int(v_y)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR TO GRAYSCALE
    img_gray = cv2.medianBlur(img_gray,3) # Removes salt and pepper noise by convolving the image with a (3,3) square kernel
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)  # Smoothens the image by convolving the with a (9,9) Gaussian filter 

    lines = lsd(img_gray)
    # Selecting required lines form all possible lines.
                # cv2.line(img,(x3,y3),(x4,y4),(0,0,255),w/2)

    sel_lines,thetas,mags,w = process(lines,img,v_x,v_y)

    sel_lines = np.array(sel_lines)

    print ("Number of lines selected are {0}".format(sel_lines.shape[0]))

    if sel_lines.shape[0] == 0:
        sys.exit("No lines found")


    theta_min = sel_lines[:,5].min()
    theta_max = sel_lines[:,5].max()

    mag_max = sel_lines[:,6].max()
    mag_min = sel_lines[:,6].min()

    tmin_ind = np.where(sel_lines==theta_min)[0][0]
    tmax_ind = np.where(sel_lines==theta_max)[0][0]
    mmin_ind = np.where(sel_lines==mag_min)[0][0]
    mmax_ind = np.where(sel_lines==mag_max)[0][0]

    L1 = line([sel_lines[tmax_ind,0],sel_lines[tmax_ind,1]],[sel_lines[tmax_ind,2],sel_lines[tmax_ind,3]])

    L2 = line([sel_lines[tmin_ind,0],sel_lines[tmin_ind,1]],[sel_lines[tmin_ind,2],sel_lines[tmin_ind,3]])

    R = intersection(L1, L2)

    if R:
        v_x,v_y = R
    else:
        sys.exit("No intersection point detected")

    innerangle = angle(sel_lines[tmax_ind],sel_lines[tmin_ind])

    mvp = np.tan(innerangle)
    mvpdeg = (innerangle*180/math.pi)

    # print 'mvpdeg is {0}'.format(innerangle*180/math.pi)

    # print 'Polar line equation value (distance) at the vanishing point is {0}'.format(pm)

    cvp = v_y - mvp*(v_x)
    yfin = img.shape[0]
    xfin = int((yfin -cvp)/mvp)


    # cv2.line(img,(sel_lines[tmax_ind,0].astype(int),sel_lines[tmax_ind,1].astype(int)),(sel_lines[tmax_ind,2].astype(int),sel_lines[tmax_ind,3].astype(int)),(255,0,0),int((sel_lines[tmax_ind,4].astype(int))/2))

    # cv2.line(img,(sel_lines[tmin_ind,0].astype(int),sel_lines[tmin_ind,1].astype(int)),(sel_lines[tmin_ind,2].astype(int),sel_lines[tmin_ind,3].astype(int)),(0,0,255),int((sel_lines[tmin_ind,4].astype(int))/2))

    cv2.line(img,(int(v_x),int(v_y)),(int(xfin),int(yfin)),(0,255,0),4)

    cv2.circle(img,(int(v_x),int(v_y)), 10, (0,255,255), 3)

    # Drawing the desired line in black

    cv2.line(img,(int(img.shape[1]/2),0),(int(img.shape[1]/2),img.shape[0]),(0,0,0),3)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", setvp)

    # while True:
    #     dontwrite = 1
    #     cv2.imshow("image",img)
    #     cv2.waitKey(1000)
    #     cv2.destroyAllWindows()
    #     break
        # k = cv2.waitKey(0) & 0xFF
            
        # if k == 27:      
        #     cv2.destroyAllWindows()

        # if k == ord('n'):
        #     break

        # if k == ord('s'):

        #     # output.write(zzz)
        #     sss = "/home/vdorbala/ICRA/Images/Test/{0}.png".format(file)
        #     cv2.imwrite(sss,img);
        #     break


    #After setting the new v_x, v_y. The same slope is considered.

    # cvp = v_y - mvp*(v_x)
    theta_m = thetam(sel_lines[tmax_ind],sel_lines[tmin_ind]) 

    # showagain = 0

    # if showagain == 1:

    #     showagain = 0

    #     yfin = img.shape[0]
    #     cv2.line(img,(int(v_x),int(v_y)),(int(xfin),int(yfin)),(0,255,0),int(w/2))

    #     cv2.circle(img,(int(v_x),int(v_y)), 10, (0,255,255), 3)

    #     #Drawing the desired line in black

    #     cv2.line(img,(int(img.shape[1]/2),0),(int(img.shape[1]/2),img.shape[0]),(0,0,0),3)

    #     cv2.imshow('image',img)
    #     k = cv2.waitKey(0) & 0xFF
                
    #     if k == 27:      
    #         cv2.destroyAllWindows()
    #     # xfin = int((yfin -cvp)/mvp)
    #     den = (xfin-v_x)
        
    #     if den == 0:
    #         den = 1

    #     theta = np.arctan((yfin-v_y)/den)
    #     # if theta>90:
    #     if theta<0:
    #         theta = -math.pi/2 - theta
    #     if theta>0:
    #         theta = math.pi/2 - theta
    #     # innerangle = math.pi/2 - theta
    #     theta_m = theta
    #     mvp = np.tan(theta_m)
    #     # print ("Angle is {}".format(innerangle*180/math.pi))
        
    #     cvp = v_y - mvp*(v_x)

    # print ("TM is {}".format(theta_m*180/math.pi))

    ptd1 = [img.shape[1]/2,0]
    ptd2 = [img.shape[1]/2,img.shape[0]]


    des_theta = 90*math.pi/180  # Because the desired line lies at the center of the image.

    des_line = [img.shape[1]/2,0,img.shape[1]/2,img.shape[0],w/2,des_theta,np.sqrt(np.square(ptd2[0] - ptd1[0]) + np.square(ptd2[1] - ptd1[1]))]

    mvp_line = [v_x,v_y,xfin,yfin,w/2,innerangle,np.sqrt(np.square(v_x - xfin) + np.square(v_y - yfin))]

    mdl= np.tan(des_theta)

    ydes = ptd1[1]
    xdes = ptd1[0]

    cdes = ydes - mdl*xdes

    v_x = v_x - (img.shape[1]/2)        ##CHANGING TO CARTESIAN COORDINATES AS GIVEN IN THE PAPER
    v_y = -(v_y - (img.shape[0]/2))

    # print (v_x)

    v_x = (v_x)*(1/5675) #Converting pixels to meters. Scale is 1m = 5675 pixels
    v_y = (v_y)*(1/5675)
    # print 'theta_m is {0}'.format(theta_m*180/math.pi)

    sel_lines = sel_lines.astype(int)
    selind = tmax_ind

    # print("Thetam is {}".format(theta_m))         

    h = 1.6  #0.47 for umich
    l = 0

    w = 0

    s = np.sin(theta_m)
    c = np.cos(theta_m)

    pm = v_x*c + v_y*s

    lambda_m = np.cos(theta_m)/h

    print ("V_X is {}".format(v_x))

    error = [[v_x],[theta_m]]

    error = np.matrix(error)

    # print 'Error is {0}\n'.format(error)
    # print (v_x)

    # Just testing the ideal case
    iderr = ([0],[0])
    iderr = np.matrix(iderr)

    le = 100*error

    # print ("le is {}".format(le))

    lemax = 0


    Jw = [[1+np.square(v_x)],[((-1)*lambda_m*l*c) + (lambda_m*w*pm) + (pm*s)]]

    Jw = np.matrix(Jw)

    # print 'Jw is {0}'.format(Jw)

    Jv = [[0],[(-1)*(lambda_m*pm)]]

    Jv = np.matrix(Jv)

    # print '\nJv is {0} \n'.format(Jv)
    vconst = 0.2

    pinv = (-1)*(np.linalg.pinv(Jw))

    # print 'Pseudo inverse is {0} \n'.format(pinv)

    fmat = le + Jv*vconst

    # print 'fmat is {0},{1},{2} \n'.format(type(fmat),fmat.shape,fmat)

    w = pinv*(fmat)

    print ('w is {0} \n'.format(w))

    w = float(w)

    return w,img


## PROCESS

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,710)
fontScale              = 2
fontColor              = (0,270,0)
lineType               = 4

max_val = 3.9251497395*0.7897025648 - 0.0967433495
min_val = -4.538150963*0.7897025648 - 0.0967433495

warray = []
PATH =  "/home/vdorbala/ICRA/Captured_Images/Captured10/"

def main():
    global warray
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

    for i in range(30):
        image = get_image(imgtopic)
        image1 = image.copy()
        image1 = cv2.flip(image,1)
        # if image.shape[2]>3:
        #     image = image[:,:,:3]
        # image =  cv2.resize(image,(227,227))

        b_channel, g_channel, r_channel, a_channel = cv2.split(image)

        image = cv2.merge((b_channel, g_channel, r_channel))
        w,img = classical(image1)
        # image = torch.from_numpy(image)
        # image = cv2.flip (image,1)
        # image1 = cv2.flip(image1,1)
        # imgno += 1

        # image = np.reshape(image,(3,227,227))

## Uncomment for neural network

        # image = image_loader(image)

        # # print ("Finished acquiring image")

        # model.eval()

        # with torch.no_grad():
        #     w = model(image)



        # print ("Before {}".format(w))

        # w = ((w.cpu().numpy().flatten()))#*0.7897025648) - 0.0967433495


        # w = w*0.7897025648 - 0.0967433495
        # print ("w original is {}".format(w))
        # w = w #+ 0.3#+ 0.35655435
        print ("w original is {}".format(w))

        if (imgno%2==0):
            # cv2.putText(image1,'{}'.format(str(-1*w)[1:-1]),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.imwrite(PATH + str(num) + ".png" ,image1)
            warray.append(w)
            num += 1
            send_vel(float(w),saber)
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

if __name__ == '__main__':
    
    ino = main()
    #print ("Warray values are : \n Max = {} \n Min = {} \n Average = {}".format(max(warray),min(warray),sum(warray)/len(warray)))
    file = open("/path/for/writing/values/obtained/","w+")
    warray = [val for sublist in warray for val in sublist]
    for no in range(len(warray)):   
        file.write("{},{}\n".format(no,warray[no]))
    #print ("Imgno is {}".format(ino))
