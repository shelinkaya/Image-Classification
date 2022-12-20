#Import required packages
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchsummary import summary 
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch.optim as optim

#read image
img = read_image('tiger.jpg')
img=Image.open("tiger.jpg") 
plt.imshow(img)


## Transformation Process
preprocess=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   mean=[0.485,0.456,0.406],
                                   std=[0.229,0.224,0.225] 
                               )
])

img_tensor=preprocess(img)
img_tensor.shape
batch=img_tensor.unsqueeze(0)
batch.shape


##Defining Pretrained model
model= models.alexnet(pretrained=True)
device="cuda" if torch.cuda.is_available() else "cpu" 
model.eval()

model.to(device)
y=model(batch.to(device))
y.shape
y_max, index=torch.max(y,1)
print(index, y_max)
#here the output will be 292.. in the next codes we will print 292 to see which class it is.

url="http://pytorch.tips/imagenet-labels"
fname="imagenet_class_labels.txt"

urllib.request.urlretrieve(url,fname)
with open("imagenet_class_labels.txt") as f:
    classes=[line.strip() for line in f.readlines()]
print(classes[292])
#output will show the class 292: 'tiger, Panthera tigris'

#now see the accuracy
prob=torch.nn.functional.softmax(y, dim=1)[0]*100
print(classes[index[0]],
     prob[index[0]].item())





