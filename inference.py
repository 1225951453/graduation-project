from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import torch
import random
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from resnext50 import create_resnext50_32x4d
from similar_to_darknet import create_my_network
from torchvision import models
from similar_to_unet import create_similar_to_unet
from sklearn.utils import shuffle

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=1,hue=0.5),
    transforms.RandomHorizontalFlip(p = 0.5), #水平翻转
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

name = {'2':"damage_by_water",'0':"health",'1':"hydropenia","3":"lack_of_light","4":"sunburn"}
chg = {"damage_by_water":u"水涝","health":u"健康","hydropenia":u"缺水","lack_of_light":u"缺光照","sunburn":u"晒伤"}
pretrained = "F://new_weight_2.pth"

from data_reader import resize_image
criterion = torch.nn.CrossEntropyLoss()

model = create_similar_to_unet(pretrained)
model.eval()

with open(r"C:/Users/WANGYONG/Desktop/internet+/classification_code/test_data.txt",'r') as f:
    lines = f.readlines()
    f.close()

lines = shuffle(lines)
corr = 0
tot = 0
with torch.no_grad():
    for i in lines:
        L = i.split(",")
        name = L[0]
        label = int(L[1])
        image = cv2.imread(name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = resize_image(img,256,256)
        img = Image.fromarray(img)
        img = preprocess(img)
        # img = torch.tensor(img,dtype=torch.float32)
        img = torch.unsqueeze(img, 0)
        predict = F.softmax(model(img),dim = 1)

        # print("pred:{}".format(predict))
        idx1 = torch.max(predict,1)[1]
        corr += (idx1 == label).sum().item()
        tot += 1
        # print("pred:{},label:{}".format(idx1.item(),label))

print(corr/tot)
