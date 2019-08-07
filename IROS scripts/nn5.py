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
import copy

import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import os
from os import path
from cnn_finetune import make_model

import random
import time

all_beginning = time.time()

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='cnn_finetune')

parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')

args = parser.parse_args()

batch_size = 8
batch_size_test = 1
epochs_num = args.epochs
epoch_inner = 0
use_gpu=0
rdeftrain = 0
rdeftest = 0

tta = []
toa = []

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=1 
    print ("CUDA Available")

print ("Batch size is {0}".format(batch_size))
print ("Batch size test is {0}".format(batch_size_test))
print ("Epoch size is {0}".format(epochs_num))
print ("This is nn5. name is Crossval")


model_ft = make_model('resnet18', num_classes=1, pretrained=True, input_size=(224, 224))#,dropout_p=0.5)
model_ft_best = copy.deepcopy(model_ft)

model_ft = model_ft.to(device)
model_ft_best = model_ft_best.to(device)

if use_gpu == 1:
    model_ft = nn.DataParallel(model_ft).cuda()
    model_ft_best = nn.DataParallel(model_ft_best).cuda()


std = [0.7897025648,8.74902560651906E-05,0.0024705311,0.7899036919,0.0060156205]
mean = [-0.0967433495,1.0000407699,-0.0001078711,0.0967841388,-0.0008226814]

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

def everything(value):
    # PATH = 'alexnet_clean.pt'
    global epoch_inner,model_ft

    epoch_inner += 1

    beginning = time.time()

    epno = 0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=model_ft.original_model_info.mean,
        #     std=model_ft.original_model_info.std),
        # transforms.ToPILImage(),
    ])


    # frun = open("values_running.csv","w+")

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

    # print(model_ft)


    # if path.exists(PATH):
    #     model_ft.load_state_dict(torch.load(PATH))

    criterion = nn.MSELoss()

    optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    totalset = Dataloader(csv_file='train_all.csv', root_dir='/users/v.dorbala/me/finall',transform=transform)
    #transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

    # trsize = int(len(totalset)*2/3)
    # tesize = len(totalset) - trsize
    # print (trsize,tesize,len(totalset))

    validation_split = 0.05
    shuffle_dataset = True
    random_seed = value

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

    trainlossarr = []
    testlossarr = []

    trainrsqarr = []
    testrsqarr = []
    # def restart_program():

    #     os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
    #     python = sys.executable
    #     os.execl(python, python, * sys.argv)


    def train(epoch):
        global rdeftrain,model_ft_best
        since = time.time()
        total_loss = 0
        total_size = 0
        traintargetarr = []
        trainoutputarr = []
        model_ft.train()

        for batch_idx, values in enumerate(train_loader):
            
            # print("Values are size {0}, {1}, {2}".format(len(values),(values['image'].shape),(values['velocity'].shape)))

            data, target, omega = values['image'], values['velocity'], values['omega']

            # print("Data type is {0}, {1}.".format(data,data.shape))

            # target = target.view(-1,4)

            omega = omega.view(-1,1)
            # print("Now target is {0}".format(target)) 

            target = omega.view(-1,1)           # CHANGE TO target.view(-1,4) for JACO

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
        # print ("RSQtrain is {}".format(rsqtrain))
        # print ("Train Loss is {}".format(total_loss/total_size))
        trainlossarr.append(total_loss/total_size)
        trainrsqarr.append(rsqtrain)
        if rsqtrain>rdeftrain:
            rdeftrain = rsqtrain
            # torch.save(model_ft.state_dict(), PATH)
            # print("Saving best value. Which is {}".format(rsqtrain))
            model_ft_best = copy.deepcopy(model_ft)

        # frun.write("{},".format(total_loss/total_size))
        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))
        #Testing

        model_ft_best.eval()

        testtargetarr = []
        testoutputarr = []

        warraytest = []
        woutputarr = []

        Jwarr = []
        Fmatarr = []
        warray = []
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

                # target = target.view(-1,4)

                omega = omega.view(-1,1)

                target = omega.view(-1,1)       # CHANGE to target.view(-1,4) for JACO

                data, target = data.to(device), target.to(device)

                # print (data.size())

                output = model_ft_best(data)
            
                testtargetarr.append(target.tolist())
                testoutputarr.append(output.tolist())
                warraytest.append(omega.tolist())

                # output = output.tolist()
                # output = [val for sublist in output for val in sublist]

                # FOR JACO

                # j = 0
                # for i in range(0,int(len(output)/4)):
                #     # Jwarr.append(float(output[j].item()))
                #     # Jwarr.append(float(output[j+1].item()))
                #     # Jmat.append(float(output[j+2].item()))
                #     # Jmat.append(float(output[j+3].item()))
                #     a = output[j]
                #     b = output[j+1]
                #     c = output[j+2]
                #     d = output[j+3]

                #     a = a*std[1] + mean[1]
                #     b = b*std[2] + mean[2]
                #     c = c*std[3] + mean[3]
                #     d = d*std[4] + mean[4]

                #     Jw = np.matrix([[a],[b]])
                #     Jmat1 = np.matrix([[c],[d]])

                #     pinv = (-1)*(np.linalg.pinv(Jw))
                #     fmat = Jmat1
                #     w = pinv*(fmat)
                #     w = torch.FloatTensor(w)
                #     woutputarr.append(w.tolist())
                #     j = j+4
                #     # print (j)


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
        # print ("RSQ for omega is {}% \n".format(rsqwarr*100))
        testrsqarr.append(rsqwarr)
        # test_loss /= len(test_loader.dataset) 

        # rsq = r2_score(tta.flatten(),toa.flatten())
        # rsq1 = r2_score(tta[::4].flatten(),toa[::4].flatten())
        # rsq2 = r2_score(tta[1::4].flatten(),toa[1::4].flatten())
        # rsq3 = r2_score(tta[2::4].flatten(),toa[2::4].flatten())
        # rsq4 = r2_score(tta[3::4].flatten(),toa[3::4].flatten())

        # print ("The Rsq percentages are: \n RSQ ALL is {} \n For 1: {}% \n For 2: {}% \n For 3: {}% \n For 4: {}%".format(rsq*100,rsq1*100,rsq2*100,rsq3*100,rsq4*100))


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # print ("\n Train-test epoch number is {}.".format(epoch))

        # if epoch % 10 == 0:
            # print("Loss at epoch {} is {}%".format(epoch,100*(fincorr/epoch)))

    # torch.save(model_ft.state_dict(), PATH)

    time_elapsed = time.time() - beginning
    print('Process at inner epoch {} complete in {:.0f}m {:.0f}s'.format(epoch_inner,time_elapsed // 60,time_elapsed % 60))

    return max(testrsqarr)



rsqarray = []
epoch = 0

nums = [x for x in range(10,100)]
random.shuffle(nums)


def initialize():
    global rdeftrain,rdeftest,tta,toa,model_ft_best,model_ft
    rdeftrain = 0
    rdeftest = 0
    tta = []
    toa = []
    model_ft = make_model('resnet18', num_classes=1, pretrained=True, input_size=(224, 224))#,dropout_p=0.5)
    model_ft_best = copy.deepcopy(model_ft)

    model_ft = model_ft.to(device)
    model_ft_best = model_ft_best.to(device)

    if use_gpu == 1:
        model_ft = nn.DataParallel(model_ft).cuda()
        model_ft_best = nn.DataParallel(model_ft_best).cuda()    


for epoch in range(0,10):
    initialize()
    value = nums[epoch]
    rsq = everything(value)
    rsqarray.append(rsq)
    print ("Currently at cross-validation epoch value of {}.\n Rsq value is {}".format(epoch+1,rsq))

time_elapsed = time.time() - all_beginning
print('FINAL PROCESS COMPLETE in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

print ("Best Rsq value was {}. \n Worst value was {}. Average is {}".format(max(rsqarray),min(rsqarray),(sum(rsqarray)/(len(rsqarray)))))

file = open("rsqvalues.csv","a+")

file.write("\nRSQ values during validation time are \n")

for i in range(len(rsqarray)):
    file.write("{},".format(rsqarray[i]))

file.close()

# t = np.arange(0,epochs_num,1)
# 
# plt.plot(t,trainrsqarr,t,testrsqarr)
# plt.xlabel("Epochs")
# plt.ylabel("R-squared value")
# plt.title("R Squared Values (Clean)")
# plt.show()

# plt.plot(t,trainlossarr,t,testlossarr)
# plt.xlabel("Epochs")
# plt.ylabel("Loss value")
# plt.title("Train and test losses (Clean)")
# plt.show()
# print ("Overall Loss on test data is {}%)".format(test_loss))