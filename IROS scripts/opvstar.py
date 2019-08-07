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

PATH = 'jaco2642_noisy_resnet1.pt'

file = open("opvstar.csv","w+")

model = make_model('resnet18', num_classes=4, pretrained=True, input_size=(227, 227))

model_ft = model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=model.original_model_info.mean,
    #     std=model.original_model_info.std),
    # # transforms.ToPILImage(),
])


# class Dataloader(Dataset):

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.data.iloc[idx, 0])

#         image = io.imread(img_name)

#         # print(img_name)

#         # image = resize(image,(250,250), anti_aliasing=True)

#         # image = torchvision.transforms.ToPILImage(image)

#         # image = torchvision.transforms.functional.to_pil_image(image)
#         # image = torchvision.transforms.functional.resize(image,(225,225))

#         # # image = torchvision.transforms.ToTensor(image)
#         # image = torchvision.transforms.functional.to_tensor(image)

#         # image = image.reshape([image.shape[2], image.shape[1], image.shape[0]])

#         # image = torch.FloatTensor(image)

#         velocity = self.data.iloc[idx, 1:].values
#         velocity = velocity.astype('float').reshape(-1, 1)

#         velocity = torch.FloatTensor(velocity)

#         sample = {'image': image, 'velocity': velocity}

#         if self.transform:
#             image1 = self.transform(sample['image'])
#             image1 = torch.FloatTensor(image1)
#             sample = {'image': image1, 'velocity': velocity}

#         # sample = data_utils.TensorDataset(sample)

#         return sample

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
        
        # print(img_name)

        image = io.imread(img_name)

        # image = resize(image,(250,250), anti_aliasing=True)

        # image = torchvision.transforms.ToPILImage(image)

        # image = torchvision.transforms.functional.to_pil_image(image)
        # image = torchvision.transforms.functional.resize(image,(225,225))

        # # image = torchvision.transforms.ToTensor(image)
        # image = torchvision.transforms.functional.to_tensor(image)

        # image = image.reshape([image.shape[2], image.shape[1], image.shape[0]])

        # image = torch.FloatTensor(image)

        velocity = self.data.iloc[idx,2:].values

        omega = self.data.iloc[idx,1]

        velocity = velocity.astype('float').reshape(1, 4)

        omega = omega.astype('float').reshape(1,1)

        velocity = torch.FloatTensor(velocity)

        omega = torch.FloatTensor(omega)

        sample = {'image': image, 'velocity': velocity, 'omega': omega}

        if self.transform:
            image1 = self.transform(sample['image'])
            image1 = torch.FloatTensor(image1)
            sample = {'image': image1, 'velocity': velocity, 'omega': omega}

        # sample = data_utils.TensorDataset(sample)

        return sample


if use_gpu == 1:
    model_ft = nn.DataParallel(model_ft).cuda()

if path.exists(PATH):
    model_ft.load_state_dict(torch.load(PATH))
else:
    sys.exit("Path not found")

totalset = Dataloader(csv_file='test_all.csv', root_dir='/users/v.dorbala/me/finall',transform=transform)

rsqmax = 0
rsqmin = 1

nums = [x for x in range(10,100)]
random.shuffle(nums)

testlossarr = []
testrsqarr = []

std = [0.7897025648,8.74902560651906E-05,0.0024705311,0.7899036919,0.0060156205]
mean = [-0.0967433495,1.0000407699,-0.0001078711,0.0967841388,-0.0008226814]

warray = []

def everything(value):
    global rsqmax,rsqmin,tta,toa,rdeftest,std,mean
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

    test_loader = torch.utils.data.DataLoader(totalset, batch_size=32,
                                            num_workers=2,sampler=valid_sampler)

    model_ft.eval()

    testtargetarr = []
    testoutputarr = []

    warraytest = []
    woutputarr = []
    warray = []

    Jwarr = []
    Fmatarr = []

# Helper function to show a batch
    # def show_landmarks_batch(sample_batched):
    #     """Show image with landmarks for a batch of samples."""
    #     images_batch, landmarks_batch = \
    #             sample_batched['image'], sample_batched['velocity']
    #     batch_size = len(images_batch)
    #     im_size = images_batch.size(2)

    #     grid = utils.make_grid(images_batch)
    #     plt.imshow(grid.numpy().transpose((1, 2, 0)))

    # for i_batch, sample_batched in enumerate(test_loader):
    #     if i_batch == 3:
    #         plt.figure()
    #         show_landmarks_batch(sample_batched)
    #         plt.axis('off')
    #         plt.title('Test Dataset Batch')
    #         plt.ioff()
    #         # plt.show()
    #         break


    with torch.no_grad():
        for batch_idx, values in enumerate(test_loader):

            data, target, omega = values['image'], values['velocity'], values['omega']

            target = target.view(-1,4)

            omega = omega.view(-1,1)

            data, target = data.to(device), target.to(device)

            # print (data.size())

            output = model_ft(data)
        
            testtargetarr.append(target.tolist())
            testoutputarr.append(output.tolist())
            warraytest.append(omega.tolist())

            output = output.tolist()
            output = [val for sublist in output for val in sublist]

            j = 0
            for i in range(0,int(len(output)/4)):
                # Jwarr.append(float(output[j].item()))
                # Jwarr.append(float(output[j+1].item()))
                # Jmat.append(float(output[j+2].item()))
                # Jmat.append(float(output[j+3].item()))
                a = output[j]
                b = output[j+1]
                c = output[j+2]
                d = output[j+3]

                a = a*std[1] + mean[1]
                b = b*std[2] + mean[2]
                c = c*std[3] + mean[3]
                d = d*std[4] + mean[4]

                Jw = np.matrix([[a],[b]])
                Jmat1 = np.matrix([[c],[d]])

                pinv = (-1)*(np.linalg.pinv(Jw))
                fmat = Jmat1
                w = pinv*(fmat)
                w = torch.FloatTensor(w)
                woutputarr.append(w.tolist())
                j = j+4
                # print (j)


    tta = [val for sublist in testtargetarr for val in sublist]
    # tta = tta[1::4]
    tta = np.array(tta)
    tta = tta.flatten()
    # toa = np.array(testoutputarr)
    toa = [val for sublist in testoutputarr for val in sublist]
    # toa = toa[1::4]
    toa = np.array(toa)
    toa = toa.flatten()

    wtarget = [val for sublist in warraytest for val in sublist]
    wtarget = np.array(wtarget)
    wtarget = wtarget.flatten()

    for idx,element in enumerate(wtarget):
        warray.append(element*std[0] + mean[0])

    wtarget = warray

    woutput = [val for sublist in woutputarr for val in sublist]
    woutput = np.array(woutput)
    woutput = woutput.flatten()

    rsqwarr = r2_score(wtarget,woutput)
    print ("RSQ for omega for epoch {} is {}% \n".format(value,rsqwarr*100))
    # test_loss /= len(test_loader.dataset) 

    rsq = r2_score(tta.flatten(),toa.flatten())
    rsq1 = r2_score(tta[::4].flatten(),toa[::4].flatten())
    rsq2 = r2_score(tta[1::4].flatten(),toa[1::4].flatten())
    rsq3 = r2_score(tta[2::4].flatten(),toa[2::4].flatten())
    rsq4 = r2_score(tta[3::4].flatten(),toa[3::4].flatten())

    # print (len(tta[::4].flatten()))
    print ("The Rsq percentages are: \n RSQ ALL is {} \n For 1: {}% \n For 2: {}% \n For 3: {}% \n For 4: {}%".format(rsq*100,rsq1*100,rsq2*100,rsq3*100,rsq4*100))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.6)

    plt.suptitle("Regression plots of the Network Output and the derived Omega", fontsize = 10)

    plt.subplot(211)
    plt.plot(toa,tta, 'o')
    plt.plot(tta,tta)
    plt.xlabel("Output")
    plt.ylabel("Target")

    plt.title("Network Output", fontsize = 8)

    plt.subplot(212)
    plt.plot(woutput,wtarget, 'o')
    plt.plot(wtarget,wtarget)
    plt.xlabel("Output")
    plt.ylabel("Target")

    plt.title("Omega Output", fontsize = 8 )

    # plt.show()

    if rsq>rsqmax:
        rsqmax = rsq

    if rsq<rsqmin:
        rsqmin = rsq

    if value%5==0:
        print("Max value so far is {}".format(rsqmax*100))
        print("Min value so far is {}\n".format(rsqmin*100))

for value in range(0,1):
    everything(value)
