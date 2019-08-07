#RUN with python3
from __future__ import division
import os
from os import path

import time
import sys

sys.path.insert(0, '/home/vdorbala/ICRA/pylsd1')

from pysabertooth import Sabertooth
import serial.tools.list_ports as port
import math

import numpy as np
from numpy import sqrt
from skimage import io, transform
from skimage.transform import rescale, resize

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pylsd.lsd import lsd

from cnn_finetune import make_model
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable


max_val = 3.9251497395*0.7897025648 - 0.0967433495
min_val = -4.538150963*0.7897025648 - 0.0967433495

warray = []
# PATH1 =  "/home/vdorbala/ICRA/videos/Captured10/"

v_const = 0.2

pipe = 0

stop_name = "stop"
move_name = "move"
rev_name = "rev"

def image_loader(image):
    
    loader = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])    
    imsize = 224
    image = loader(image)
    image = torch.FloatTensor(image)
    # sample = {'image': image}
    image = image.unsqueeze(0) # For VGG
    # print("Image shape is {}".format(image.shape))
    image = Variable(image, requires_grad=True)
    return image.cuda()

def cnn(image):
    global warray

    image = cv2.imread("/home/vdorbala/ICRA/Captured_Images/Captured22/10.png")
    PATH = 'direct2642_noisy_resnet2.pt' #direct2642_noisy_resnet2.pt, direct2642_noisy.pt
    # If RoS is being used
    # listener()
    # model,saber,imgtopic = initialize()
    model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(227, 227))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu=1 
        print ("CUDA Available")
    else:
        device = torch.device("cpu")

    if use_gpu == 1:
        model = nn.DataParallel(model).cuda()

    if path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
    else:
        print("Did not find Model!")

    model = model.to(device)

    print ("\n Initialization complete")

    # print ("q value is {}".format(q.get()))
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((227,227)),
    #     transforms.ToTensor(),
    # ])
    print (image.shape)
    if image.shape[2]>3:
        image = image[:,:,:3]

    image = image_loader(image)

    model.eval()

    with torch.no_grad():
        w = model(image)

    # print ("Before {}".format(w))

    w = ((w.cpu().numpy().flatten()))
    w = w*0.7897025648 - 0.0967433495
    return w

def cuberoot(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

def inv_solver(w):
    # img = cv2.imread("/home/vdorbala/ICRA/Captured_Images/Captured22/10.png")
    # w,img1 = classical(img)
    # print ("w original is {}".format(w))
    lambda_1 = 100
    w = float(w[0])
    # print ("Angular Velocity is {}".format(w))

    C = np.divide((-1*w), 2*lambda_1)

    D = sqrt((np.square(w)/(4*(np.square(lambda_1)))) + (1/27))

    A = cuberoot(C + D)
    B = cuberoot(C - D)

    solution = A + B
    # print(solution)
    return solution


def main():
    img = cv2.imread("/home/vdorbala/ICRA/Captured_Images/Captured22/10.png")
    w = cnn(img)
    print ("w original is {}".format(w))
    vp = inv_solver(w)
    print("VP solution is {}".format(vp))


if __name__ == '__main__':
    
    ino = main()