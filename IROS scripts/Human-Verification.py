# For Human Verification on Unreliable images.
# Opens up an image where the user has to click in the direction where the wheelchair might turn.
# Compares the input to the prediction of the CNN and calculates a precision score.

from __future__ import print_function, division

import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
from skimage.transform import rescale, resize
import torch.nn.functional as F     
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
import math
import sys
# from sklearn.metrics import r2_score

import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import time
import os
from os import path
from cnn_finetune import make_model

import random
import time

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')



if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=1 
    print ("CUDA Available")

PATH = 'direct2642_noisy_resnet2.pt'

file = open("opvstar.csv","w+")

model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(224, 224))

model_ft = model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=model.original_model_info.mean,
    #     std=model.original_model_info.std),
    # # transforms.ToPILImage(),
])

v_x = 0
v_y = 0
showagain = 0

corr = 0
pres = 0
presno = 0

class Dataloader(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])

        image = io.imread(img_name)

        # print(img_name)

        # image = resize(image,(250,250), anti_aliasing=True)

        # image = torchvision.transforms.ToPILImage(image)

        # image = torchvision.transforms.functional.to_pil_image(image)
        # image = torchvision.transforms.functional.resize(image,(225,225))

        # # image = torchvision.transforms.ToTensor(image)
        # image = torchvision.transforms.functional.to_tensor(image)

        # image = image.reshape([image.shape[2], image.shape[1], image.shape[0]])

        # image = torch.FloatTensor(image)

        velocity = self.data.iloc[idx, 1]
        velocity = velocity.astype('float').reshape(-1, 1)

        velocity = torch.FloatTensor(velocity)

        sample = {'image': image, 'velocity': velocity}

        if self.transform:
            image1 = self.transform(sample['image'])
            image1 = torch.FloatTensor(image1)
            sample = {'image': image1, 'velocity': velocity}

        # sample = data_utils.TensorDataset(sample)

        return sample


if use_gpu == 1:
    model_ft = nn.DataParallel(model_ft).cuda()

if path.exists(PATH):
    model_ft.load_state_dict(torch.load(PATH))

model_ft = model.to(device)

criterion = nn.MSELoss()

optimizer = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

totalset = Dataloader(csv_file='hard1.csv', root_dir='/home/vdorbala/ICRA/Images/Hard1',transform=transform) # Change to Hard1

rsqmax = 0
rsqmin = 1

nums = [x for x in range(10,100)]
random.shuffle(nums)

def setvp(event,x,y,flags,param):
    global v_x,v_y,showagain

    showagain = 1

    if event == cv2.EVENT_LBUTTONDOWN:
        print ("Left is {},{}".format(x,y))

    elif event == cv2.EVENT_LBUTTONUP:
        v_x = x
        v_y = y

validation_split = 1
shuffle_dataset = True
random_seed=nums[0]

# Creating data indices for training and validation splits:
dataset_size = len(totalset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# print ("Number of images in test set are {}".format(len(valid_sampler)))

train_loader = torch.utils.data.DataLoader(totalset, batch_size=1,
                                        num_workers=2,sampler=valid_sampler)

# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, landmarks_batch = \
#             sample_batched['image'], sample_batched['velocity']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)

#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
coords = []

def onclick(event):
    global v_x, v_y,coords
    ix, iy = event.xdata, event.ydata
    # print ("x = {}, y = {}".format(ix, iy))

    v_x = ix
    v_y = iy
    coords.append(ix)

    # if len(coords) == 1:
    #     fig.canvas.mpl_disconnect(cid)

    return v_x

trainlossarr = []
testlossarr = []

trainrsqarr = []
testrsqarr = []

total_loss = 0
total_size = 0
traintargetarr = []
trainoutputarr = []
model_ft.train()

num = 0
for batch_idx, values in enumerate(train_loader):
    v_x = 0
    # print("Values are size {0}, {1}, {2}".format(len(values),(values['image'].shape),(values['velocity'].shape)))

    data, target = values['image'], values['velocity']

    # print("Data type is {0}, {1}.".format(data,data.shape))

    target = target.view(-1,1)

    # print("Now target is {0}".format(target)) 

    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    output = model_ft(data)

    loss = criterion(output, target)

    images_batch = values['image']
    batch_size1 = len(images_batch)
    im_size = images_batch.size(2)

    fig = plt.figure()

    plt.ion()
    plt.show()

    grid = utils.make_grid(images_batch)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    plt.pause(2)
    plt.close()

    output = (output.detach().cpu().numpy().flatten())*0.7897025648 - 0.0967433495

    if v_x == 0:
        sys.exit("No click detected. Click to the left of the image for left and right for right")

    if np.sign(v_x-113)!=np.sign(output):
        print("Correct prediction!")
        print (-1*output)
        corr +=1
    else:
        print("Wrong prediction!")
        pres += abs(output)
        print (-1*output)
        presno +=1

    num += 1
    # loss.backward()
    
    # optimizer.step()
print ("Total number of correctly predicted images is {} of {}. Percentage is {}".format(corr,num,corr/num))
print ("Precision score is {}/{} which is {} percentage".format(pres,presno,pres/presno))
