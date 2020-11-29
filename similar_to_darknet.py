import torch
import torch.nn as nn
import torchsummary as ts
import torch.nn.functional as F
import pysnooper
import numpy as np

class similar_to_darknet(nn.Module):

    def __init__(self):

        super(similar_to_darknet,self).__init__() #channel ???feature map
        #bottleneck
        self.conv1 = nn.Conv2d(3,512,kernel_size=3,stride=2,padding = 1) #,padding=(1,0)
        self.b1 = nn.BatchNorm2d(512)
        self.a1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512,256,kernel_size=1,stride = 1) #padding=(1,0)
        self.b2 = nn.BatchNorm2d(256)
        self.a2 = nn.ReLU(inplace = True)
        self.conv3 = nn.Conv2d(256,512,kernel_size=3,stride = 1,padding = 1) #,padding=(1,0),
        self.b3 = nn.BatchNorm2d(512)
        self.a3 = nn.ReLU(inplace = True)

        self.conv4 = nn.Conv2d(512,256,kernel_size=(3,1),stride=1) # 1x1
        self.b4 = nn.BatchNorm2d(256)
        self.a4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256,128,kernel_size=(1,3),stride=1) # padding
        self.b5 = nn.BatchNorm2d(128)
        self.a5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128,64,kernel_size=3,stride=1)
        self.b6 = nn.BatchNorm2d(64)
        self.a6 = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(64*124*124,1000)
        self.l2 = nn.Linear(1000,128)
        self.l3 = nn.Linear(128,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):

        x = self.conv1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.conv3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.conv4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.conv5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.conv6(x)
        x = self.b6(x)
        x = self.a6(x)
        print(np.shape(x))
        x = x.view(x.size(0),-1)
        print(np.shape(x))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

def create_my_network(pretrained):

    model = similar_to_darknet()
    # pretrained = None
    if pretrained:
        if isinstance(pretrained, str):

            model_dict = model.state_dict()
            pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    return model

model = create_my_network(None)
ts.summary(model, (3, 256, 256))
