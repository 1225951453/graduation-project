from renext50 import create_resnext50_32x4d
from data_reader import preprocess
import numpy as np
import os
import torch.nn as nn
import torch
from PIL import Image
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from torch import optim
import pandas as pd
import time
import random
from sklearn.utils import shuffle
from data_reader import reader

model = create_resnext50_32x4d()   #Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
model.eval()
# image,label = reader()
# x_train,x_test,y_train,y_test = train_test_split(image,label,test_size=0.3)
# dataset_train = TensorDataset(torch.from_numpy(np.array(x_train,dtype=np.float32)), torch.from_numpy(np.array(y_train,dtype=np.float32)))
# dataset_test = TensorDataset(torch.from_numpy(np.array(x_test,dtype=np.float32)),torch.from_numpy(np.array(y_test,dtype=np.float32)))
lr = 0.001
optimizer = optim.Adam(model.parameters(),lr=lr)
lr_sc = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 2, ) #T_mul = 1
criterion = nn.CrossEntropyLoss()
batch_size = 16
epoches = 30
with open("C:/Users/WANGYONG/Desktop/internet+/classification_code/data.txt",'r') as f:
    data_li = f.readlines()
    f.close()

data_li = shuffle(data_li)
# max_batch = len(dataset_train)//batch_size
epoch_size = len(data_li) // batch_size
# Iter = iter(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
# Iter = iter(dataset_train)
model.train()

for epoch in range(epoches):
    loss = 0.0
    acc = 0.0
    prev_time = time.time()
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{(epoches - 0)}', postfix=dict,mininterval=0.3) as pbar:
        for iteration in range(1,epoch_size+1):
            # data_list = data_li[iteration-1:16*iteration]
            Iter = reader(data_li,batch_size)
            images, labels = next(Iter)
            # labels = labels.view(-1,1)
            images = images.to('cpu')
            labels = labels.to('cpu')
            optimizer.zero_grad()
            predict = model(images).to('cpu')
            tmp_loss = criterion(predict,labels)
            loss = loss + tmp_loss.item()
            tmp_loss.backward()
            optimizer.step()

            prediction = torch.max(predict.data, 1)[1]
            # prediction = torch.argmax(predict)
            train_correct = (prediction == labels).sum()
            ##?????train_correct???longtensor???????float
            # print(train_correct.type())
            train_acc = (train_correct.float()) / batch_size
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'train_loss': loss ,
                                # "miou": epoch_miou / (iteration + 1),
                                "acc":train_acc.float(),
                                'lr': lr,
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

        lr_sc.step()

        print('Finish Validation')
        # print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (
                loss / (epoch_size + 1)))  # , val_loss / (epoch_size_val + 1)

