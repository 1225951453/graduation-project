import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imageio
# import imgaug as ia
# from imgaug import augmenters as iaa

# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.2),
#     # iaa.Sometimes(
#     #     0.5,
#     #     iaa.GaussianBlur(sigma=(0, 0.5))
#     # ),
#     iaa.Affine(rotate=(-100,100)),
#     iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
#     iaa.AddToHueAndSaturation((-60, 60)),  # change their color
#     iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
#     iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)  # set large image areas to zero
# ], random_order=True)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def read_image(filename, resize_height=None, resize_width=None, normalization=True):
    # rgb_image = imageio.imread(filename)
    # print("img_type:{}".format(type(rgb_image)))
    bgr_image = cv2.imread(filename)
    if bgr_image is None:
        print("Warning:???:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # ???????????
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    return bgr_image

def resize_image(image, resize_height, resize_width):

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ?BGR??RGB
    image = cv2.resize(image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    return image

def save_image(image_path, rgb_image, toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image, dtype=np.uint8)
    if len(rgb_image.shape) == 2:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=256, resize_width=256, repeat=1):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        # self.cp = transforms.CenterCrop(224)
        self.jit = transforms.ColorJitter(brightness=1,hue=0.5)
        self.radomRotation = transforms.RandomRotation(10) #随即旋转
        self.flip = transforms.RandomHorizontalFlip(p = 0.5) #水平翻转
        self.ver = transforms.RandomVerticalFlip(p=0.5) #垂直翻转

        self.toTensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name,label = self.image_label_list[index]
        # print("image_name,label:{}".format(image_name,label))
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
        img = self.data_preproccess(img)
        label = np.array(label)
        return img, label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(',')
                image = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((image, labels))
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        image = read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        data = resize_image(data, 256, 256)
        data = Image.fromarray(data)
        # data = self.cp(data)
        data = self.radomRotation(data)
        # data = self.ver(data)
        data = self.flip(data)
        data = self.jit(data)
        data = self.toTensor(data)
        # data = self.norm(data)
        return data

filename = r"C:/Users/WANGYONG/Desktop/internet+/classification_code/data.txt"
image_dir = r"C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo"

# filename = r"F:/plant/plant_data.txt"
# image_dir = r"F:/Nonsegmented"

from torchvision import datasets, models, transforms
from similar_to_unet import create_similar_to_unet
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
# from data_reader import TorchDataset

dataset = TorchDataset(filename,image_dir)
dataloders = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
Len = dataset.len
# Len1 = dataset.__len__()
# pretrained = False
pretrained = r"F:/new_weight_2.pth"
lr = 0.001
criterion = nn.CrossEntropyLoss()
batch_size = 32
model = create_similar_to_unet(pretrained)
model.eval()
# dataloders = {"train":torch.utils.data.DataLoader(image_datasets['train'],batch_size=2,shuffle=True,num_workers=4)}
# optimizer = torch.optim.Adam(model.parameters(),lr=lr)
optimizer = torch.optim.SGD(model.parameters(),lr = lr,momentum=0.9,weight_decay=1e-4)
lr_sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 2) #T_mul = 1
epoches = 10

#冻结骨干train10个epoch
for param in model.modules():
    if not isinstance(param,nn.Linear):
        # print(param)
        param.requires_grad = False

for e in range(epoches):
    running_loss = 0.0
    running_corrects = 0.0
    model.train(True)
    lr_sc.step()
    print("epoch:{}".format(e+1))
    # inputs, labels = (dataloders)
    start = time.time()
    for data in dataloders:
        inputs, labels = data
        if torch.cuda.is_available():
            labels = labels.squeeze(1)
            inputs = torch.tensor(inputs.cuda())
            labels = torch.tensor(labels.cuda())
        else:
            labels = labels.squeeze(1)
            # inputs,labels = Variable(inputs),Variable(labels)
            inputs, labels = torch.tensor(inputs,dtype=torch.float32), torch.tensor(labels,dtype=torch.int64)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        print("preds:{}".format(preds))
        # loss = F.cross_entropy(outputs,labels)
        loss = criterion(outputs, labels)
        # if phase == 'train':
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == labels).item()
    print("time:{}".format(time.time()-start))
    print("loss:{},acc:{}".format(running_loss/Len,running_corrects/Len))

#解冻骨干 trian 10个epoch
for param in model.modules():
    if not isinstance(param,nn.Linear):
        # print(param)
        param.requires_grad = True

for e in range(epoches):
    running_loss = 0.0
    running_corrects = 0.0
    model.train(True)
    lr_sc.step()
    print("epoch:{}".format(e+1))
    # inputs, labels = (dataloders)
    for data in dataloders:
        inputs, labels = data
        if torch.cuda.is_available():
            labels = labels.squeeze(1)
            inputs = torch.tensor(inputs.cuda())
            labels = torch.tensor(labels.cuda())
        else:
            labels = labels.squeeze(1)
            # inputs,labels = Variable(inputs),Variable(labels)
            inputs, labels = torch.tensor(inputs,dtype=torch.float32), torch.tensor(labels,dtype=torch.int64)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # print("preds:{}".format(preds))
        # loss = F.cross_entropy(outputs,labels)
        loss = criterion(outputs, labels)
        # if phase == 'train':
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == labels).item()

    print("loss:{},acc:{}".format(running_loss/Len,running_corrects/Len))


print("save the model ................")
torch.save(model.state_dict(),"F://new_weight_2.pth")
