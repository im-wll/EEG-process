#将Keras版本的EEG conv改成pytorch版本
#参考：https://github.com/tom-beer/deep-sleep-mind/blob/master/networks.py
 #https://github.com/HypoX64/candock/blob/master/models/cnn_1d.py
import torch
from torch import nn
import torch.nn.functional as F

#多个通道的信号一起处理
class ConvNet(nn.Module):
    def __init__(self, inchannel, num_classes):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv1d(inchannel, out_channels=64, kernel_size=7, stride=1, padding=3,bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace = True),  
                nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, out_channels=128, kernel_size=7, stride=1, padding=3,bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),  
                nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(         
                nn.Conv1d(128, 256, 7, 1, 3, bias=False),
                nn.BatchNorm1d(256),    
                nn.ReLU(inplace = True),                     
                nn.MaxPool1d(2),               
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.out = nn.Linear(64, num_classes)  


    def forward(self, x):
#        print('x.shape',x.shape) #(batch,22,1000)
        x = self.conv1(x)  #(batch,64,500)
#        print('x1.shape',x.shape)
        x = self.conv2(x)  #(batch,128,250)
#        print('x2.shape',x.shape)
        x = self.conv3(x)  #(batch,256,125)
#        print('x3.shape',x.shape)
        x = self.avgpool(x) #(batch,256,1)
#        print('x4.shape',x.shape)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

#--------------------------------------------------
#对每一个通道的信号进行conv处理
class OneChannelConv(nn.Module):
    def __init__(self, inchannel):
        super(OneChannelConv,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv1d(inchannel, out_channels=4, kernel_size=7, stride=1, padding=3,bias=False),
                nn.BatchNorm1d(4),
                nn.ReLU(inplace = True),  
                nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
                nn.Conv1d(4, out_channels=8, kernel_size=7, stride=1, padding=3,bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace = True),  
                nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(         
                nn.Conv1d(8, 16, 7, 1, 3, bias=False),
                nn.BatchNorm1d(16),    
                nn.ReLU(inplace = True),                     
                nn.MaxPool1d(2),               
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
         
    def forward(self, x):
#        print('x.shape',x.shape) #(batch,1,1000)
        x = self.conv1(x)  #(batch,4,500)
#        print('x1.shape',x.shape)
        x = self.conv2(x)  #(batch,8,250)
#        print('x2.shape',x.shape)
        x = self.conv3(x)  #(batch,16,125)
#        print('x3.shape',x.shape)
        x = self.avgpool(x) #(batch,16,1)
#        print('x4.shape',x.shape)
        x = x.view(x.size(0), -1)  #(batch,16)
       
        return x


#基础分类器
class BaseClassifierNet(nn.Module):
    def __init__(self, num_classes, feature_len):
        super().__init__()
        self.feature_len = feature_len
        self.fc1 = nn.Linear(self.feature_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # batch_size x C x  N
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    

#对每一个通道的信号进行conv处理
class MultiChannelConv(nn.Module):
    def __init__(self, inchannel,num_classes, elecnodes):
        super(MultiChannelConv,self).__init__()
        self.onechannel_feature = OneChannelConv(inchannel)
        self.elecnodes =elecnodes
        self.feature_len = self.elecnodes*16
        self.fc1 = nn.Linear(self.feature_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
#        print('x.shape',x.shape) #(batch,22,1000)
        batch_feature = torch.zeros(x.size(0),1)
        batch = x.size(0)
        
        for node in range(self.elecnodes):
           eachNode_input = torch.unsqueeze(x[:,node,:], 1) #每一个节点通道的原始数据
#           print('eachNode_input.shape',eachNode_input.shape) #(64,1,1000)
           eachNode_feature = self.onechannel_feature(eachNode_input) #每一个节点通道的特征
#           print('eachNode_feature.shape',eachNode_feature.shape) #(64,16)
           batch_feature = torch.cat([batch_feature,eachNode_feature],dim=1)  #(64,352)

        batch_feature = batch_feature[:,1:]
#        print('batch_feature.shape',batch_feature.shape)
#        batch_feature =batch_feature.view(batch,self.elecnodes,-1)
#        print('batch_feature2.shape',batch_feature.shape)
        batch_feature =batch_feature.view(batch,-1)
        fc1 = self.fc1(batch_feature)
        fc2 = self.fc2(fc1)
        return fc2

