from __future__ import print_function, division
import os
import re
import torch
import pandas as pd
import cv2
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import torch.optim as optim
import torch.nn.init as weight_init

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print ("CUDA AVAILABLE!")
    # y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    # x = x.to(device)                       # or just use strings ``.to("cuda")``
    # z = x + y
    # print(z)
    # print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

print("torch version: %s"%torch.__version__)

# plt.ion()   # interactive mode
# angular_vel = pd.read_csv('cmu2.csv')

# n = 65
# img_name = angular_vel.iloc[n, 0]

velocity = []

DATA_ROOT = '/users/v.dorbala/me/fin2'

# Hyper Parameters
num_epoch = 10               # train the training data n times, to save time, we just train 1 epoch

class Dataload(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.angular_vel = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.angular_vel)

    def __getitem__(self, idx):

        # print('This is {0}'.format(self.angular_vel.iloc[idx, 0]))

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])

        # print ('name is {0}'.format(img_name))

        velocity = self.angular_vel.iloc[idx, 1].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)

        image = io.imread(img_name)

        self.angular_vel.iloc[idx,1] = re.findall(r"\d+\.\d+",self.angular_vel.iloc[idx,1])


        velocity = self.angular_vel.iloc[idx, 1]

        velocity = float(velocity[0])


        # print (velocity)

        sample = {'image': image, 'velocity': velocity}

        if self.transform:
            sample = self.transform(sample)

        return sample


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, velocity = sample['image'], sample['velocity']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for velocity because for images,
        # x and y axes are axis 1 and 0 respectively
        velocity = velocity * [new_w / w, new_h / h]

        return {'image': img, 'velocity': velocity}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, velocity = sample['image'], sample['velocity']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        velocity = velocity - [left, top]

        return {'image': image, 'velocity': velocity}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, velocity = sample['image'], sample['velocity']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'velocity': torch.from_numpy(velocity)}


trainset = Dataload(csv_file='cmu2.csv', root_dir='/users/v.dorbala/me/fin2/',transform=None)#transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = Dataload(csv_file='vps1.csv', root_dir='/users/v.dorbala/me/testfin/',transform=None)#transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 50)
#         self.fc3 = nn.Linear(50, 1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net().to(device)

net = torchvision.models.alexnet(pretrained=False)



# optimization scheme can be 'sgd', 'RMSProp', 'Adam', 'Adadelta', 'Adagrad'
optimization_scheme = "Adam"
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

if optimization_scheme == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
elif optimization_scheme == 'RMSProp':
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0)
elif optimization_scheme == "Adadelta":
     optimizer = optim.Adadelta(net.parameters(), lr=learning_rate, weight_decay=0)
elif optimization_scheme == "Adam":
     optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)
elif optimization_scheme == "Adagrad":
     optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=0)


for epoch in range(num_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        print (type(data))
        # get the inputs
        inputs, labels = data

        print (inputs[0])

        print (type(inputs),type(labels))

        # print ("Shapes are {0},{1}. Type is {3} {4}".format(len(inputs),len(labels),type(inputs),type(labels)))
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

 # Quantitative Analysis
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))  # Qualitative Analysis
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# print images
imshow(torchvision.utils.make_grid(images.cpu()))
print('GroundTruth: ', ' '.join('%4s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))