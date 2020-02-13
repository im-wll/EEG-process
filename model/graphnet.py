"""
Created on Sun Sep 29 15:45:28 2019
@author: wanglinlin
"""
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchvision
import torch.nn.functional as F
import numpy as np


def get_adj_matrix(x,k_neighbor=4):
    """
    计算每个节点之间的距离,返回最近的8个邻居的位置,得到邻接矩阵
    generate N x N node distance matrix 生成n x n节点距离矩阵,计算每个节点之间的距离
    param x: batch_size x  Node x channel (N nodes with C feature dimension) #(batch,22,16)
    return: batch_size x N x N  邻接矩阵
    """
    batch_size = x.size(0)
    node = torch.squeeze(x) #torch.squeeze() 这个函数主要对数据的维度进行压缩,去掉维数为1的的维度 
#    print('node.shape:',node.shape) #batch*node*channel

    if batch_size == 1:
        node = torch.unsqueeze(node, 0)#torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为1的维度.
    node_transpose = torch.transpose(node, dim0=1, dim1=2) #返回输入矩阵input的转置。交换维度dim0和dim1 #batch*channel*node
#    print('node_transpose.shape:',node_transpose.shape)
    
    node_inner = torch.matmul(node,node_transpose) 
    node_inner = -2 * node_inner #节点内部: -2*(node)T*(node)
#    print('node_inner.shape',node_inner.shape) #(batch,node,node)
    
    node_square = torch.sum(node ** 2, dim=2, keepdim=True) #节点的平方和
#    print('node_square.shape',node_square.shape)
    node_square_transpose = torch.transpose(node_square, dim0=1, dim1=2) #节点平方和的转置
#    print('node_square_transpose.shape',node_square_transpose.shape)
    #计算每个节点间的欧式距离,得到batch*node*node
    distance_matrix=node_square + node_square_transpose + node_inner 
#    print('distance_matrix:',distance_matrix)
#    print('distance_matrix.shape',distance_matrix.shape)

    batch_size = distance_matrix.size(0) 
    num_point = distance_matrix.size(1) #节点个数
    _, nn_idx = torch.topk(distance_matrix, k_neighbor, dim=2, largest=False) #沿给定dim维度返回输入张量input中 k 个最小值,返回距离最小的index
    nn_idx=nn_idx.type(torch.LongTensor)
#    print('nn_idx',nn_idx)
    
    #邻接矩阵，nn_idx设1，其余设0
    adj_matrix= torch.zeros(batch_size,num_point,num_point)

    for i in range (batch_size):
        for j in range(num_point):
            for k in range(k_neighbor):
                index=nn_idx[i][j][k]
                adj_matrix[i][j][index]=1
                           
#    print(adj_matrix)
#    print('adj_matrix.shape',adj_matrix.shape)
                
    #乘以度矩阵D,等于gen_adj,这只适用于每个节点邻接矩阵的度一样的情况
    adj_matrix = torch.mul(adj_matrix,(1/k_neighbor)) 
    return adj_matrix  #batch*node*node


#对邻接矩阵进行处理，乘以度矩阵D
def gen_adj(A):
#    print('A',A)
#    print('A.shape',A.shape)
    batch=A.size(0)
#    print('batch',batch)
    num_point=A.size(1)
    Digree=torch.zeros(batch,num_point,num_point)
    for i in range(batch):
        D = torch.pow(A[i].sum(1).float(), -0.5)
#        print('D',D)
#        print('D.shape',D.shape)
        D = torch.diag(D)
#        print('D2',D)
#        print('D2.shape',D.shape)
        Digree[i]=D
#    print('Digree',Digree)
#    print('Digree.shape',Digree.shape)    
    adj = torch.matmul(A, D).t()
#    print('adj1.shape',adj.shape)
    adj = adj.permute(1, 0, 2) 
#    print('adj2.shape',adj.shape)
    adj = torch.matmul(adj,D)
#    adj = torch.matmul(torch.matmul(A, D).t(),D)
    print('adj3',adj)
    print('adj3.shape',adj.shape)
    return adj


#-------------------------方法2，参考pygcn,需要先得到邻接矩阵A-----------------------------
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features #16
        self.out_features = out_features #8
            
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #(16,8)
        if bias:
            self.bias = Parameter(torch.FloatTensor(1,1,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) #input*weight error! (batch,22,16) *(16,32)->(batch,22,32)
#        print('support.shape',support.shape)
#        adj=adj.to('cuda')
        output = torch.matmul(adj, support) #adj* (input*weight) (22)
#        print('output',output)
#        print('output.shape',output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN3(nn.Module):
    def __init__(self, nclass, k_neighbor,inchannel=1,elecnodes=22,dropRate=0.5):
        super(GCN3, self).__init__()
        self.prefeature = MultiChannelConv( inchannel,elecnodes) 

        self.gc1 = GraphConvolution(16,32)
        self.gc2 = GraphConvolution(32,64)
        self.gc3 = GraphConvolution(64,128)

        self.fc1 = nn.Linear(128,64)
        self.cls_layers =  nn.Linear(64,nclass)
        self.k_neighbor = k_neighbor
        self.droprate=dropRate

    def forward(self, x):
#        print('input.shape',x.shape)
        x = self.prefeature(x) #(batch,22,16)
#        x = self.feature2(x)     
#        print('feature.shape',x.shape)  
        x = x.view(x.size(0), x.size(1), -1) #batch*node*channel
        adj= get_adj_matrix(x,self.k_neighbor)
#        print('adj.shape',adj.shape)
        
        x1 = self.gc1(x, adj) #error!
        x1 = F.relu(x1) 
        x2 = self.gc2(x1, adj)
        x2 = F.relu(x2) 
        x3 = self.gc3(x2, adj) #batch*node*channel
        x3 = F.relu(x3)  
        
        x3 = x3.permute(0, 2, 1) #batch*channel*node
        x3 = x3.view(x3.size(0), x3.size(1), -1)
        x3 = x3.mean(dim=-1)

        fc1 = self.fc1(x3)
        output=self.cls_layers(fc1)
#        print('output.shape',output.shape)
        
        return output


#对每一个通道的信号进行conv处理
class MultiChannelConv(nn.Module):
    def __init__(self, inchannel,elecnodes):
        super(MultiChannelConv,self).__init__()
        self.onechannel_feature = OneChannelConv(inchannel)
        self.elecnodes =elecnodes
       
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
        batch_feature =batch_feature.view(batch,self.elecnodes,-1) #(batch,22,16)
#        print('batch_feature2.shape',batch_feature.shape) 
        
        return batch_feature


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