import sys
import os
import numpy as np
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import re
import random

import shutil
from shutil import copyfile

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


f = open("hard.csv","r+")

lis = [line.split() for line in f]
imgarr = []
valarr = []

for i,x in enumerate(lis):
    imgno,val1,val2,val3,val4,val5 = x[0].split(',')
    imgarr.append(imgno)
    valarr.append(val1)


# # FOR SELECTING IMAGES
# maxthresh = 1
# minthresh = 0
# maxno = 0
# revalno = 0
# print (len(valarr))
# for i in range(len(imgarr)):
#     if abs(float(valarr[i]))<maxthresh and abs(float(valarr[i]))>minthresh:
#         maxno += 1
#         path = "/home/vdorbala/ICRA/Images/finall2/" + imgarr[i]
#         print (path,valarr[i])
#         img = cv2.imread(path)
#         reval = "/home/vdorbala/ICRA/Images/reval/" + imgarr[i]
#         hard = "/home/vdorbala/ICRA/Images/Hard/" + imgarr[i]

#         cv2.imshow("image",img)
#         k = cv2.waitKey(0) & 0xFF
                
#         if k == 27:      
#             cv2.destroyAllWindows()
#         if k ==ord('n'):
#             cv2.imwrite(reval,img)
#             revalno += 1
#         if k ==ord('c'):
#             cv2.imwrite(hard,img)

# percent = (float(maxno)/float(len(imgarr)))*100

# print ('{}/{} images i.e, {}% are below threshold.'.format(maxno,len(imgarr),percent))

# For Removing images from csv file
m = 0
remaining = []
len1 = len(lis)
PATH = '/home/vdorbala/ICRA/Images/Hard1'
for file in sorted(os.listdir(PATH), key=numericalSort):
    if file in imgarr:
        remaining.append(lis[imgarr.index(file)])
        lis[imgarr.index(file)] = (0,0)
        m += 1

lis = [z for z in lis if z != (0,0)]

print ("Removed {} elements.".format(len1 - len(lis)))

# for item in lis:
#   o.write("{}\n".format(str(item)))
# o1 = open('easy.csv','w+')
o2 = open('hard1.csv','w+')
# o1 = open('all_all1.csv', 'w+')  #For Trainset
# o2 = open('all_all2.csv', 'w+') #For testset
# o1 = open('train_all.csv', 'w+')  #For Trainset including noisy
# o2 = open('test_all.csv', 'w+') #For testset including noisy - 528 images

lis = [val for sublist in lis for val in sublist]

# for item in lis: 
#     # print (str(item))
#     o1.write('{}\n'.format(item))

remaining = [val for sublist in remaining for val in sublist]

for item in remaining:
    o2.write('{}\n'.format(item))

f.close()

#For randomly seperating out test set

# PATH_TEST = '/users/v.dorbala/me/testset/'
# # PATH_TRAIN = '/users/v.dorbala/me/trainset3023/'
# PATH_ROOT = '/users/v.dorbala/me/finall/'

# for file in sorted(os.listdir(PATH_ROOT), key = numericalSort):
#     test_split = 0.1
#     shuffle_dataset = True

#     nums = [x for x in range(10,100)]
#     random.shuffle(nums)

#     random_seed=nums[0]

#     dataset_size = len(lis)

#     indices = list(range(dataset_size))
#     split = int(np.floor(test_split * dataset_size))
#     if shuffle_dataset :
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_indices, test_indices = indices[split:], indices[:split]

# # print (test_indices)

# for idx,i in enumerate(test_indices):
#     shutil.copy(PATH_ROOT + str(imgarr[i]),PATH_TEST)

# print ("Done Test")

# for idx,i in enumerate(train_indices):
#     shutil.copy(PATH_ROOT + str(imgarr[i]),PATH_TRAIN)

# # #Extract images from csv file to path

# # PATH = '/home/vdorbala/ICRA/Images/extracted/'
# PATH_ROOT = '/home/vdorbala/ICRA/Images/finall2/'

# for i in range(len(imgarr)):
#     shutil.copy(PATH_ROOT + str(imgarr[i]),PATH)
