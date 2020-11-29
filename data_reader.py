from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import torch
import random
import cv2

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def add_gaussian(image,Scale):
    gaussian_noise_img = gaussian(image, scale=Scale)
    return gaussian_noise_img

def gaussian(src, scale):
    gaussian_noise_img = np.copy(src)
    # gaussian_noise_img = src
    noise = np.random.normal(0, scale, size=(3,src.shape[1], src.shape[2])) #??
    add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #???????
    add_noise_and_check += noise
    # add_noise_and_check = add_noise_and_check.astype(np.int16)
    gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
    return gaussian_noise_img

preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
#
# def preprocess(img):
#     img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
#     return img/255.
#
# def reader(data_li,batch_size):
#     image = []
#     label = []
#     img_num = 0
#     label_num = 0
#
#     for pth in data_li:
#         Img_pth = pth.split(",")[0]
#         # lab = int(pth.split(",")[1].strip())
#         lab = float(pth.split(",")[1].strip())
#         img = cv2.imread(Img_pth)
#         # img = Image.open(Img_pth).convert('RGB')
#         img = preprocess(img)
#         # img = add_gaussian(np.array(img), 0.01)
#         # img.transpose(1,2,0)
#         # img = Image.fromarray(img)
#         # img.show()
#         # img = torch.unsqueeze(torch.tensor(img), dim=0).float()
#
#         image.append(img)
#         label.append(lab)
#         del img
#         del lab
#         if len(image) == batch_size:
#             image = np.array(image,dtype=np.float64)
#             image = image.transpose(0,3,1,2)
#             # image = np.array(image)
#             # image = np.squeeze(image)
#             # img = np.
#             # image = torch.unsqueeze(torch.tensor(image), dim=0)
#             # label = torch.unsqueeze(torch.tensor(label),dim = 0)
#             label = np.array(label,dtype=np.long)
#             yield torch.from_numpy(image),torch.from_numpy(label)
#             # yield image,torch.from_numpy(label).long()
#             # image.clear()
#             # label.clear()
#             del image
#             del label

def smooth_labels(y_true, label_smoothing,num_classes):
    num = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    return torch.tensor(num,dtype = torch.float64)

# import imageio
# import imgaug as ia
# from imgaug import augmenters as iaa
# ia.seed(4)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(filename, resize_height=None, resize_width=None, normalization=True):
    bgr_image = cv2.imread(filename)
    if bgr_image is None:
        print("Warning:???:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # ???????????
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # ?BGR??RGB
    return rgb_image / 255.0

def resize_image(image, resize_height, resize_width):

    image = cv2.resize(image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    return image

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=256, resize_width=256, repeat=1):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.toTensor = transforms.ToTensor()
        # self.radomRotation = transforms.RandomRotation(10)
        # self.normalize=transforms.Normalize()

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
        data = resize_image(data,256,256)
        data = seq.augment_image(data)
        # data = self.radomRotation(data)
        data = self.toTensor(data)
        return data

