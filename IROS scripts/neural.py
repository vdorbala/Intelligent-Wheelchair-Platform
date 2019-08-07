import os
import sys
import numpy as np
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import numpy as np
from skimage import io, transform
from skimage.transform import rescale, resize
from cnn_finetune import make_model
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable

PATH = 'direct2642_noisy_resnet2.pt'
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu=1 
    print ("CUDA Available")
else:
    device = torch.device("cpu")

model = make_model('resnet18', num_classes=1, pretrained=True)
# print (model)
if use_gpu == 1:
    model = nn.DataParallel(model).cuda()

if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))


loader = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])

def image_loader(image):
    
    imsize = 224
    image = loader(image)
    image = torch.FloatTensor(image)
    # sample = {'image': image}
    image = image.unsqueeze(0) # For VGG
    image = Variable(image, requires_grad=True)
    return image.cuda()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,710)
fontScale              = 2
fontColor              = (0,0,0)
lineType               = 3

warray = []
file1 = open("/home/vdorbala/ICRA/cap21.csv","w+")
file2 = open("/home/vdorbala/ICRA/captured21_n.csv","w+")


for subdir, dirs, files in os.walk('/home/vdorbala/ICRA/Captured_Images/Captured21/'):
    files.sort()
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

            img = io.imread(filepath)
            # img = resize(img,(227,227))

            if img.shape[2]>3:
                img = img[:,:,:3]

            # print (img.shape)
            # img = cv2.resize(img,(227,227))
            # img = cv2.resize(img,(224,224)) # For Resnet18
            img1 = img.copy()
            # img = np.reshape(img,(3,227,227))
            # print(img.shape)
            img = image_loader(img)

            # data_load = torch.utils.data.DataLoader(img)

            # print (img.shape)
            # print ("Finished acquiring image")

            model.eval()

            with torch.no_grad():
                w = model(img)

            w = ((w.cpu().numpy().flatten()))#*0.7897025648) - 0.0967433495

            w = w*0.7897025648 - 0.0967433495
            # print ("w original is {}".format(w))
            w = w #+ 0.3#+ 0.35655435
            # print ("w is {}".format(w))

            # warray.append(w)
            file2.write("{},{}\n".format(file,float(w)))
            # cv2.putText(img1,'{}'.format(str(w)[1:-1]),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            # # cv2.imwrite(PATH + str(imgno) + ".png" ,image1)
            # cv2.imshow('Window',img1)
            # k = cv2.waitKey() & 0xFF
            # if k == 27:
            #     cv2.destroyAllWindows()
            # if k == ord('s'):
            #     sss = "{},{}\n".format(file1,w)
            #     file1.write(sss)

# warray = [val for sublist in warray for val in sublist]
# for no in range(len(warray)):   
#     file.write("{},{}\n".format(no,warray[no]))