# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# class simpleconv3(nn.Module):
#     def __init__(self):
#         super(simpleconv3,self).__init__()
#         self.conv1 = nn.Conv2d(3, 12, 3, 2)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(12, 24, 3, 2)
#         self.bn2 = nn.BatchNorm2d(24)
#         self.conv3 = nn.Conv2d(24, 48, 3, 2)
#         self.bn3 = nn.BatchNorm2d(48)
#         self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
#         self.fc2 = nn.Linear(1200 , 128)
#         self.fc3 = nn.Linear(128 , 2)
#     def forward(self , x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         #print "bn1 shape",x.shape
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = x.view(-1 , 48 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# data_dir = '../../../../datas/head/'
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomSizedCrop(48),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#     ]),
#     'val': transforms.Compose([
#         transforms.Scale(64),
#         transforms.CenterCrop(48),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#     ]),
# }
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x]) for x in ['train', 'val']}
# dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                              batch_size=16,
#                                              shuffle=True,
#                                              num_workers=4) for x in ['train', 'val']}
#
# torch.init.xavier_uniform(self.conv1.weight)init.constant(self.conv1.bias, 0.1)
#
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         xavier(m.bias.data)
#     net = Net()
#     net.apply(weights_init)
#
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train(True)
#             else:
#                 model.train(False)
#                 running_loss = 0.0
#                 running_corrects = 0.0
#             for data in dataloders[phase]:
#                 inputs, labels = data
#                 if use_gpu:
#                     inputs = Variable(inputs.cuda())
#                     labels = Variable(labels.cuda())
#                 else:
#                     inputs, labels = Variable(inputs), Variable(labels)
#
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#
#                 running_loss += loss.data.item()
#                 running_corrects += torch.sum(preds == labels).item()
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects / dataset_sizes[phase]
#
#             if phase == 'train':
#                 writer.add_scalar('data/trainloss', epoch_loss, epoch)
#                 writer.add_scalar('data/trainacc', epoch_acc, epoch)
#             else:
#                 writer.add_scalar('data/valloss', epoch_loss, epoch)
#                 writer.add_scalar('data/valacc', epoch_acc, epoch)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#     writer.export_scalars_to_json("./all_scalars.json")
#     writer.close()
#     return model
# import torch
# if torch.cuda.is_available():
#     pth = "dsfsdaf"
#     print(pth)
#
# else:
#     print("no!")


# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # ???????
# plt.grid()  # ????
# plt.show()
# from renext50 import create_resnext50_32x4d
# import torch
# import numpy as np
# pretrained = "F:/leaf_ckpt_0.0012094378471374512_1504.0.pth"
# model = create_resnext50_32x4d(5,pretrained)
# model_dict = model.state_dict()

# for k,v in model_dict.items():
#     print("ori_k.shape:{},ori_v.shape:{}".format(np.shape(k),np.shape(v)))
#     print("ori_k:{},ori_v:{}".format(k,v))
#
# pretrained_dict = torch.load(pretrained,map_location=torch.device('cpu'))
#
# for k,v in pretrained_dict.items():
#     print("k.shape:{},v.shape:{}".format(np.shape(k),np.shape(v)))
#     print("k:{},v:{}".format(k,v))
# {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}

# import torch
# import torch.nn as nn
#
# class Darknet(nn.Module):
#     def __init__(self):
#         super(Darknet).__init__()
#         self.block_list = [1,2,4,4,8]
#         self.channels = [32,64,128,256,512,1024]
#         self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1)
#         self.b1 = nn.BatchNorm2d(32)
#         self.a1 = nn.Relu(inplace = True)
#
#         self.res_block1 = self._make_layer([32,64],self.block_list[0])
#         self.res_block2 = self._make_layer([64,128],self.block_list[1])
#         self.res_block3 = self._make_layer([128,256],self.block_list[2])
#         self.res_block4 = self._make_layer([256,512],self.block_list[3])
#         self.res_block5 = self._make_layer([512,1024],self.block_list[4])

    # def _make_layers(self,channels,blocks):
    #     res_block = []
    #     res_block.append(nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2,padding=1))
    #     res_block.append(nn.BatchNord2d(channels[1]))
    #     res_block.append(nn.Relu(inplace = True))


# model = Darknet()
# import torch
# arr = torch.zeros_like([0,3])
# print(type(arr))

# for i in model.parameters:
#     print(i)

# a = 1
# print(type(a))
# print(a.type(float))

# import torch.nn as nn
# import torch.functional as F

# FloatTensor = F.cuda.FloatTensor #if x.is_cuda else torch.FloatTensor
# LongTensor = F.cuda.LongTensor #if x.is_cuda else torch.LongTensor
# FloatTensor = torch.FloatTensor
# LongTensor = torch.LongTensor
# bs = 32
# in_w = in_h = 446
#
# x = torch.linspace(0, in_w - 1, in_w).repeat(in_w,0)#.reshape() #.type(FloatTensor)
# print(x)
# print(x.shape)

# FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
#
# # ????????????????
# grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1) #.repeat(int(32 * 3 / 3), 1, 1) #.view(x.shape).type(FloatTensor)
# grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
#     int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)


# ????????????????
# grid_x = .repeat(in_w, 1).repeat(
#     int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
# grid_y = F.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
#     int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

# import os
# p = "C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/"
# for i in os.listdir(p):
#     p1 = os.path.join(p , i)
#     idx = 1
#     for j in os.listdir(p1):
#         # if j.split(".")[1] == 'bmp':
#         name = str(idx) + '.jpg'
#         os.rename(p1 + "/" + j,p1 + "/" + name)
#         idx = idx + 1

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def read_image(filename, resize_height=None, resize_width=None, normalization=True):
#     bgr_image = cv2.imread(filename)
#     if bgr_image is None:
#         print("Warning:???:{}", filename)
#         return None
#     if len(bgr_image.shape) == 2:  # ???????????
#         print("Warning:gray image", filename)
#         bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # ?BGR??RGB
#     return rgb_image / 255.0
#
# def resize_image(image, resize_height, resize_width):
#
#     image = cv2.resize(image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
#     return image
#
# def save_image(image_path, rgb_image, toUINT8=True):
#     if toUINT8:
#         rgb_image = np.asanyarray(rgb_image, dtype=np.uint8)
#     if len(rgb_image.shape) == 2:
#         bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
#     else:
#         bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(image_path, bgr_image)
#
# import torch
# from torch.autograd import Variable
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
#
# class TorchDataset(Dataset):
#     def __init__(self, filename, image_dir, resize_height=256, resize_width=256, repeat=1):
#
#         self.image_label_list = self.read_file(filename)
#         self.image_dir = image_dir
#         self.len = len(self.image_label_list)
#         self.repeat = repeat
#         self.resize_height = resize_height
#         self.resize_width = resize_width
#
#         self.toTensor = transforms.ToTensor()
#
#     def __getitem__(self, i):
#         index = i % self.len
#         # print("i={},index={}".format(i, index))
#         image_name,label = self.image_label_list[index]
#         # print("image_name,label:{},{}".format(image_name,label))
#         image_path = os.path.join(self.image_dir, image_name)
#         img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
#         img = self.data_preproccess(img)
#         label = np.array(label)
#         return img, label
#
#     def __len__(self):
#         if self.repeat == None:
#             data_len = 10000000
#         else:
#             data_len = len(self.image_label_list) * self.repeat
#         return data_len
#
#     def read_file(self, filename):
#         image_label_list = []
#         with open(filename, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 # rstrip?????????????(??\n?\r?\t?' '???????????????)
#                 content = line.rstrip().split(',')
#                 image = content[0]
#                 labels = []
#                 for value in content[1:]:
#                     labels.append(int(value))
#                 image_label_list.append((image, labels))
#         return image_label_list
#
#     def load_data(self, path, resize_height, resize_width, normalization):
#
#         image = read_image(path, resize_height, resize_width, normalization)
#         return image
#
#     def data_preproccess(self, data):
#         data = resize_image(data,256,256)
#
#         # data = self.radomRotation(data)
#         data = self.toTensor(data)
#         return data
# #
# filename = r"C:/Users/WANGYONG/Desktop/internet+/classification_code/data.txt"
# image_dir = r"C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo"
#
# from torchvision import datasets, models, transforms
# from similar_to_unet import create_similar_to_unet
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # data_dir = "C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/"
#
# dataset = TorchDataset(filename,image_dir)
# dataloders = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
# # Len = dataset.len
# pretrained = False
# # pretrained = "F://similar_to_unet_cifar10_weight_10.pth"
# # pretrained = "F://new_weight.pth"
#
# lr = 0.001
#
# criterion = nn.CrossEntropyLoss()
# batch_size = 32
# # epoches = 5
#
# # data_dir = "C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/"
# # image_datasets = {'train':datasets.ImageFolder(data_dir,data_transforms['train'])}
# model = create_similar_to_unet(pretrained)
# model.eval()
# # dataloders = {"train":torch.utils.data.DataLoader(image_datasets['train'],batch_size=2,shuffle=True,num_workers=4)}
# optimizer = torch.optim.Adam(model.parameters(),lr=lr)
# lr_sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 2) #T_mul = 1
# epoches = 15
#
# for e in range(epoches):
#     running_loss = 0.0
#     running_corrects = 0.0
#     model.train(True)
#     lr_sc.step()
#     print("e:{}".format(e+1))
#     # inputs, labels = (dataloders)
#     for data in dataloders:
#         inputs, labels = data
#         if torch.cuda.is_available():
#             inputs = torch.tensor(inputs.cuda())
#             labels = torch.tensor(labels.cuda())
#         else:
#             labels = labels.squeeze(1)
#             # inputs,labels = Variable(inputs),Variable(labels)
#             inputs, labels = torch.tensor(inputs,dtype=torch.float32), torch.tensor(labels,dtype=torch.int64)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         print("preds:{}".format(preds))
#         # loss = F.cross_entropy(outputs,labels)
#         loss = criterion(outputs, labels)
#         # if phase == 'train':
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.data.item()
#         running_corrects += torch.sum(preds == labels).item()
#
#     print(running_loss/Len,running_corrects/Len)
# print("save the model ................")
# torch.save(model.state_dict(),"F://new_weight_tmp_2.pth")

# import os
# import cv2
# img = r"C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/0/1.jpg"
#
# img = cv2.imread(img)
# img = cv2.resize(img,(256,256))
# img = cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)
# img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
#
# cv2.imshow("img",img)
# cv2.imwrite("img_linear.jpg",img)
# # cv2.imwrite()
# cv2.waitKey(2500)
# cv2.destroyWindow("img")

# import os
# pth = r"F:/Nonsegmented/"
# idx = 0
# with open( "F:/plant_label.txt",'w') as f:
#     for i in os.listdir(pth):
#         # for j in os.listdir(pth + i):
#         s = i + "," + str(idx)
#         f.write(s + "\n")
#
#         idx = idx +

# import os
# pth = r"C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/4/"
#
# idx = 201
# for i in os.listdir(pth):
#     src_name = pth + i
#     dst_name = pth + str(idx) + '.jpg'
#     os.rename(src_name,dst_name)
#     idx = idx + 1

# from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import KMeans
# from matplotlib import pyplot
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import silhouette_samples
# import pandas as pd
#
# # X, _ = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# X = pd.read_csv(r"C:/Users/WANGYONG/Desktop/internet+/data/tmp_data.csv")
#
# for i in range(2,16):
#     model = KMeans(n_clusters=i,random_state=0)
#     clusters_ = model.fit(X)
#     score = silhouette_score(X,clusters_.labels_)
#     print("cluser:{},score:{}".format(i,score))

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import cv2
# ia.seed(4)
#
pth = r"C:/Users/WANGYONG/Desktop/internet+/data/new_data_lvluo/4/201.jpg"
# img = cv/.imshow()
# # img_pil = Image.open(pth)
# img = imageio.imread(pth)
img = cv2.imread(pth)
# img = img/255.0
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
img = Image.fromarray(img)
img = img.transpose()
# ia.imshow(img)
#
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.2),
#     iaa.Sometimes(
#         0.5,
#         iaa.GaussianBlur(sigma=(0, 0.5))
#     ),
#     iaa.Affine(rotate=(-100,100)),
#     iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
#     iaa.AddToHueAndSaturation((-60, 60)),  # change their color
#     iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
#     iaa.CoarseDropout((0.01, 0.1), size_percent=0.01)  # set large image areas to zero
# ], random_order=True)
#
# # rotate = iaa.Affine(rotate=(-100,100))
# image_aug = seq.augment_image(img)
#
# print("Augmented:")
# ia.imshow(image_aug)










# 检索唯一群集
#     clusters = unique(clusters_.labels_)
#     # 为每个群集的样本创建散点图
#     for cluster in clusters:
#     # 获取此群集的示例的行索引
#         row_ix = where(clusters_.labels_ == cluster)
#     # 创建这些样本的散布
#         pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
#     # 绘制散点图
#         pyplot.show()

