from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np
import torch
import random

mean=[0.485, 0.456, 0.406] #,(0.485, 0.456, 0.406),(0.485, 0.456, 0.406)
std=[0.229, 0.224, 0.225] #
# normalize = transforms.Normalize(
#
# )
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(10),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    # normalize
])

num_class = 5
num = 1


# os.listdir(data_pth)

def reader(data_li,batch_size):
    image = []
    label = []
    img_num = 0
    label_num = 0
    # Dir = os.listdir(data_pth)
    # for v in Dir:
    #     for var in os.listdir(os.path.join(data_pth,v)):
    #         num = 0
    # img_pth = os.path.join(data_pth, v, var)
    # if var.split(".")[1] == 'jpg':
    for pth in data_li:
        Img_pth = pth.split(",")[0]
        lab = int(pth.split(",")[1].strip())
        img = Image.open(Img_pth).convert('RGB')
        img = preprocess(img)
        img = torch.unsqueeze(img, dim=0).float()
                # num = num + 1
                # img = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
        image.append(np.array(img))
        label.append(np.array(lab))
        # num = num + 1
        if len(image) == batch_size:
            image = np.array(image,dtype=np.float32)
            image = np.squeeze(image)
            # image = image.squeeze(image)(0)
            # image = image[:, :, :, ::-1]
            # image = image.transpose(0,3,1,2) #cv2??

            # image = image / 255*batch_size-1
            label = np.array(label)
            yield torch.from_numpy(image),torch.from_numpy(label).long()
            image.clear()
            label.clear()
                # img_num = img_num + 1
            # else:
            #     with open(img_pth,'r',encoding='utf-8') as f:
            #         line = f.readline()
            #         label.append([int(line[-1].strip())])
            #         num = num + 1
            #         f.close()
                # label_num = label_num + 1

        # if img_num != label_num:
        #     print(v)
    # return image,label
