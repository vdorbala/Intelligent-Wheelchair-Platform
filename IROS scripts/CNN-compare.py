# Training CNN using a Random Seed each time.

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
from sklearn.metrics import r2_score

import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import time
import os
from os import path
from cnn_finetune import make_model

import random
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=1 
    print ("CUDA Available")

PATH = 'direct2642_noisy_resnet4.pt'

file = open("opvstar.csv","w+")

model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(224, 224))

model_ft = model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model.original_model_info.mean,
        std=model.original_model_info.std),
    # # transforms.ToPILImage(),
])


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

        # print(image.shape)

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

totalset = Dataloader(csv_file='test_all.csv', root_dir='/users/v.dorbala/me/finall3',transform=transform)

rsqmax = 0
rsqmin = 1

nums = [x for x in range(10,100)]
random.shuffle(nums)

def everything(value):
    global rsqmax,rsqmin
    validation_split = 1 
    shuffle_dataset = True
    random_seed=nums[value]

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

    test_loader = torch.utils.data.DataLoader(totalset, batch_size=1,
                                            num_workers=2,sampler=valid_sampler)

    model_ft.eval()

    testtargetarr = []
    testoutputarr = []

    with torch.no_grad():
        for batch_idx, values in enumerate(test_loader):

            data, target = values['image'], values['velocity']

            target = target.view(-1,1)

            data, target = data.to(device), target.to(device)

            image = np.array(values['image'])
            # print(image.shape)
            image = image[0,0,:,:]
            output = model_ft(data)
            # image = transforms.ToPILImage()(image)
            # print(target.item())
            # plt.text(0.5, 0.5, str(target), horizontalalignment='center', verticalalignment='center')
            # plt.imshow(image)
            # plt.pause(2)

            # print (target)
            # print ("Test Target is {0}. \n Output is {1}.".format(target,output))
            # print ("Target type is {0}. \n Output type is {1}.".format(target.type,output.type))
            # print ("Target shape is {0}. \n Output shape is {1}.".format(target.shape,output.shape))

            # print (target, output)

            # loss = criterion(output, target).item()

            testtargetarr.append(target.tolist())
            testoutputarr.append(output.tolist())


    tta = [val for sublist in testtargetarr for val in sublist]
    tta = np.array(tta)
    tta = tta.flatten()
    # toa = np.array(testoutputarr)
    toa = [val for sublist in testoutputarr for val in sublist]
    toa = np.array(toa)
    toa = toa.flatten()

    # file.write("\n Target values during test \n")

    # for i in range(len(tta)):
    #     file.write("{},".format(str(tta[i])))

    # file.write("\n Output values during test \n")

    # for i in range(len(toa)):
    #     file.write("{},".format(str(toa[i])))

    # file.close()

    # print ("Printing to file done")
    plt.plot(tta,toa , 'o')
    plt.plot(tta,tta)
    plt.xlabel("Target")
    plt.ylabel("Output")

    plt.title("Target vs Output values")

    rsq = r2_score(tta,toa)
    print ("The Rsq percentage for epoch {} is {}%".format(value,rsq*100))

    if rsq>rsqmax:
        rsqmax = rsq
        plt.show()


    if rsq<rsqmin:
        rsqmin = rsq

    if value%10==0:
        print("Max value so far is {}".format(rsqmax))
        print("Min value so far is {}\n".format(rsqmin))

for value in range(0,1):
    everything(value)
