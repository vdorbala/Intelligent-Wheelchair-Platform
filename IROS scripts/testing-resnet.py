from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F		
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os


use_gpu = torch.cuda.is_available()

#################################################


model_conv = torchvision.models.resnet34(pretrained=True)

#print(model_conv)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

print(model_conv)
#print('-'*50)
#params = list(model_conv.parameters())
#print(params)
if use_gpu:
	model_conv = nn.DataParallel(model_conv).cuda()
model_conv.load_state_dict(torch.load('/users/sairam.tabibu/results/resnet-34-new2/resnet_wei1.pth'))


#################################################

data_dir = '/Neutron3/Datasets/med_img'
#data_dir = '/Users/user/Desktop/medical_images'
resize = [224,224]

data_transforms = {
        'test': transforms.Compose([
            #Higher scale-up for inception
            transforms.Resize(max(resize)),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.620, 0.446, 0.594], [0.218, 0.248, 0.193])
        ]),
    }


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
											 shuffle=False)
			  for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes




###################################################t


def train_model(model, num_epochs):
	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
	
		# Each epoch has a training and vali dation phase 'train',
		for phase in ['test']:
			print(phase)
			model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
			running_corr = 0
			k=1
			iteration =0
			step =0		# Iterate over data.
			for data in dataloaders[phase]:
	
				# get the inputs
				inputs, labels = data
				#idx=list(map(int,idx.numpy().squeeze()))
				#if iteration==step:
				#	for idx_e in idx:
				#		print(image_datasets[phase].imgs[idx_e])
				#	step = step + 30	

				#iteration = iteration + 1

				# wrap them in Variable
				if use_gpu:
					if phase == 'train':
						inputs = Variable(inputs.cuda())
						labels = Variable(labels.cuda())
					else:
						inputs = Variable(inputs.cuda(), volatile=True)
						labels = Variable(labels.cuda(), volatile=True)
				else:
					inputs = Variable(inputs)
					labels = Variable(labels)

				outputs = model(inputs)	

				_, preds = torch.max(outputs.data, 1)
				#print("-----------------")
				#print(preds)
				#print(loss)	

				if(k<=30):
					if(k<=20):
						print(k)
						running_corrects += torch.sum(preds == labels.data)
						f = open("classes1.txt","a+")
						f.write(str(torch.sum(preds == labels.data)) + "\n" )
						f.close()
					k=k+1			

				if(k==31):
					perc = running_corrects
					if(running_corrects > 800):
						running_corr += 1
						f = open("classes.txt","a+")
						f.write("correct" + "-" + str(perc) + "\n" )
						f.close()
					else:
						f = open("classes.txt","a+")
						f.write("False" + "-" + str(perc) + "\n" )
						f.close() 
					running_corrects = 0
					k=1		

				

			
			epoch_acc = (running_corr*1.0) / (dataset_sizes[phase]/3000)

			print('{} Acc: {:.4f}'.format(
				phase, epoch_acc))
			
				
			
		print()

	time_elapsed = time.time() - since
	print('Testing complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))



	return model


#######################################################



model_ft = train_model(model_conv,num_epochs=1)



