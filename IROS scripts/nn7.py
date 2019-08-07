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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='cnn_finetune')

parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')


PATH = 'direct2642_noisy_resnet_new.pt'

thresh = 0.05

rdeftrain = 0
rdeftest = 0

beginning = time.time()

epno = 0
use_gpu=0

tta = []
toa = []

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=1 
    print ("CUDA Available")

batch_size = 8
batch_size_test = 64
epochs_num = args.epochs

print ("Batch size is {0}".format(batch_size))
print ("Batch size test is {0}".format(batch_size_test))
print ("Epoch size is {0}".format(epochs_num))
print ("This is nn7. name is 1")

model = make_model('resnet18', num_classes=1, pretrained=True, input_size=(224, 224))


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model.original_model_info.mean,
        std=model.original_model_info.std),
    # transforms.ToPILImage(),
])


# frun = open("values_running.csv","w+")

class Dataloader(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

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
#################################################

# velocity_frame = pd.read_csv("/users/v.dorbala/me/vpscmu.csv")

# n = 65
# img_name = velocity_frame.iloc[n, 0]
# velocity = velocity_frame.iloc[n, 1:].as_matrix()
# velocity = velocity.astype('float').reshape(-1, 1)

# print('Image name: {}'.format(img_name))
# print('Velocity shape: {}'.format(velocity.shape))
# print('Velocity value: {}'.format(velocity[:1]))
# model_conv = torchvision.models.alexnet(pretrained=True)

# num_features = 1

# num_ftrs = model_conv.classifier[6].in_features
# model_conv.classifier[6] = nn.Linear(num_ftrs, num_features)

# for param in model_conv.classifier[6].parameters():
#   param.requires_grad = True

print(model)

model_ft = model.to(device)

if use_gpu == 1:
    model_ft = nn.DataParallel(model_ft).cuda()

if path.exists(PATH):
    model_ft.load_state_dict(torch.load(PATH))

criterion = nn.MSELoss()

optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

totalset = Dataloader(csv_file='train_all.csv', root_dir='/users/v.dorbala/me/finall',transform=transform)
# print ("Size of the dataset is {}".format(len(totalset)))
#transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

# trsize = int(len(totalset)*2/3)
# tesize = len(totalset) - trsize
# print (trsize,tesize,len(totalset))

validation_split = 0.1
shuffle_dataset = True

nums = [x for x in range(10,100)]
random.shuffle(nums)

random_seed=nums[int(time.time())%90]

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

# trainset, testset = torch.utils.data.random_split(totalset,[trsize,tesize])
# print (len(trainset),len(testset))
# train_sampler = SubsetRandomSampler(trainset)
# test_sampler = SubsetRandomSampler(testset)

# print (type(train_sampler))

# trainset = Dataloader(csv_file='trainall.csv', root_dir='/users/v.dorbala/me/fin3/',transform=transform)#transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

train_loader = torch.utils.data.DataLoader(totalset, batch_size=batch_size,
                                          num_workers=2,sampler=train_sampler)
# testset = Dataloader(csv_file='vps1.csv', root_dir='/users/v.dorbala/me/testfin/',transform=transform)#transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

test_loader = torch.utils.data.DataLoader(totalset, batch_size=batch_size_test,
                                        num_workers=2,sampler=valid_sampler)


# Helper function to show a batch
# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, landmarks_batch = \
#             sample_batched['image'], sample_batched['velocity']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)

#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

#     # for i in range(batch_size):
#     #     plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
#     #                 landmarks_batch[i, :, 1].numpy(),
#     #                 s=10, marker='.', c='r')

# for i_batch, sample_batched in enumerate(train_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['velocity'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.title('Train Dataset Batch')
#         plt.ioff()
#         plt.show()
#         break


# for i_batch, sample_batched in enumerate(test_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['velocity'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.title('Test Dataset Batch')
#         plt.show()
#         break
# testset = torchvision.datasets.CIFAR10(root='./Cifar10', train=False,
#                                        download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

trainlossarr = []
testlossarr = []

trainrsqarr = []
testrsqarr = []
# def restart_program():

#     os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
#     python = sys.executable
#     os.execl(python, python, * sys.argv)


def train(epoch):
    global rdeftrain
    since = time.time()
    total_loss = 0
    total_size = 0
    traintargetarr = []
    trainoutputarr = []
    model_ft.train()

    for batch_idx, values in enumerate(train_loader):
        
        # print("Values are size {0}, {1}, {2}".format(len(values),(values['image'].shape),(values['velocity'].shape)))

        data, target = values['image'], values['velocity']

        # print("Data type is {0}, {1}.".format(data,data.shape))

        target = target.view(-1,1)

        # print("Now target is {0}".format(target)) 

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model_ft(data)

        # print ("Target is {0}. \n Output is {1}.".format(target.numpy(),output.numpy()))

        # print(batch_idx)

        # rsq = r2_score(target.numpy(),output.numpy())
        
        traintargetarr.append(target.tolist())
        trainoutputarr.append(output.tolist())

        loss = criterion(output, target)

        total_loss += loss.item()

        total_size += 1

        # print ("Data size is {}. \n Total size is {}".format(data.size(0),total_size))

        
        loss.backward()
        
        optimizer.step()

        # print (batch_idx % args.log_interval)

        # if batch_idx % args.log_interval == 10:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), total_loss / total_size))
        # if math.isnan(total_loss) == True:
        #     sys.exit('Loss has gone to NaN. Doubling the batch size.')
            # restart_program()

    # tta = np.array(testtargetarr)
    tta = [val for sublist in traintargetarr for val in sublist]
    tta = np.array(tta)
    # toa = np.array(testoutputarr)
    toa = [val for sublist in trainoutputarr for val in sublist]
    toa = np.array(toa)
    # print (tta.flatten(),toa.flatten())
    rsqtrain = r2_score(tta.flatten(),toa.flatten())

    # vconst = 0.2
    # pinv = (-1)*(np.linalg.pinv(Jw))
    # fmat = le + Jv*vconst
    # w = pinv*(fmat)

    print ("RSQtrain is {}".format(rsqtrain))
    trainlossarr.append((total_loss/total_size))
    trainrsqarr.append(rsqtrain)
    if rsqtrain>rdeftrain:
        rdeftrain = rsqtrain
        # model_ft_best = copy.deepcopy(model_ft)
        torch.save(model_ft.state_dict(), PATH)
    # frun.write("{},".format(total_loss/total_size))
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))


def test():
    global tta,toa,rdeftest
    # model_ft.load_state_dict(torch.load(PATH))
    model_ft.eval()
    test_loss = 0
    total_loss = 0
    total_size = 0

    testtargetarr = []
    testoutputarr = []

    with torch.no_grad():
        for batch_idx, values in enumerate(test_loader):

            data, target = values['image'], values['velocity']

            target = target.view(-1,1)

            data, target = data.to(device), target.to(device)

            # print (data.size())

            output = model_ft(data)

            # print ("Test Target is {0}. \n Output is {1}.".format(target,output))
            # print ("Target type is {0}. \n Output type is {1}.".format(target.type,output.type))
            # print ("Target shape is {0}. \n Output shape is {1}.".format(target.shape,output.shape))

            loss = criterion(output, target)

            total_loss += loss.item()

            total_size += 1
        
            testtargetarr.append(target.tolist())
            testoutputarr.append(output.tolist())

            # print ("Loss is {}".format(loss))
            # pred = output.data.max(1, keepdim=True)[0]

            # correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    # tta = np.array(testtargetarr)
    tta = [val for sublist in testtargetarr for val in sublist]
    tta = np.array(tta)
    # toa = np.array(testoutputarr)
    toa = [val for sublist in testoutputarr for val in sublist]
    toa = np.array(toa)

    rsqtest = r2_score(tta.flatten(),toa.flatten())
    print ("RSQtest is {}\n".format(rsqtest))

    # test_loss /= len(test_loader.dataset)    
    testlossarr.append(total_loss/total_size)
    testrsqarr.append(rsqtest)
    # frun.write("{},".format(test_loss))



for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    print ("\n Epoch number is {}.".format(epoch))

    # if epoch % 10 == 0:
        # print("Loss at epoch {} is {}%".format(epoch,100*(fincorr/epoch)))

# torch.save(model_ft.state_dict(), PATH)

time_elapsed = time.time() - beginning
print('Process complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


# trace = dict(x=[1:args.epochs], y=testlossarr, mode="markers+lines", type='custom',
#              marker={'color': 'red', 'symbol': 104, 'size': "10"}, name='1st Trace')

trainlossarr = np.array(trainlossarr)
testlossarr = np.array(testlossarr)

trainrsqarr = np.array(trainrsqarr)
testrsqarr = np.array(testrsqarr)

file = open("direct2642_noisy_resnet_new.csv","w+")

file.write("Train values are \n")

for i in range(len(trainlossarr)):
    file.write("{},".format(trainlossarr[i]))

file.write("\n Test values are \n")

for i in range(len(testlossarr)):
    file.write("{},".format(testlossarr[i]))

file.write("\n Train R-squared values are \n")

for i in range(len(trainrsqarr)):
    file.write("{},".format(trainrsqarr[i]))

file.write("\n Test R-squared values are \n")

for i in range(len(testrsqarr)):
    file.write("{},".format(testrsqarr[i]))

file.close()

t = np.arange(0,epochs_num,1)

plt.plot(t,trainlossarr,t,testlossarr)
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.title("Train and test losses (Normal)")
plt.show()
# print ("Overall Loss on test data is {}%)".format(test_loss))


# viz.line(X=None, Y=None, win=win, name='delete this', update='remove')


# layout = dict(title="Testing Loss", xaxis={'Epoch': 'x1'}, yaxis={'Loss': 'x2'})

# vis._send({'data': win, 'layout': layout, 'win': 'mywin'})
# resize = [224,224]    
# data_transforms = {
#         'test': transforms.Compose([
#             #Higher scale-up for inception
#             transforms.Resize(max(resize)),
#             #transforms.RandomHorizontalFlip(),
#             #transforms.CenterCrop(max(resize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.620, 0.446, 0.594], [0.218, 0.248, 0.193])
#         ]),
#     }

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# class_names = image_datasets['train'].classes

# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'test']:
#           print (phase)
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()  # Set model to training mode
#             else:
#                 model.test()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25)