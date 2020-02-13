#https://www.cnblogs.com/RoseVorchid/p/12197841.html
# 导入工具包
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#定义网络模型：
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # 全连接层
        # 此维度将取决于数据中每个样本的时间点数。
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)


    def forward(self, x):
#        print('x0.shape',x.shape)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
#        print('x1.shape',x.shape)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
#        print('x2.shape',x.shape)

        # Layer 2
        x = self.padding1(x)
#        print('x3.shape',x.shape)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
#        print('x4.shape',x.shape)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
#        print('x5.shape',x.shape)

        # Layer 3
        x = self.padding2(x)
#        print('x6.shape',x.shape)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
#        print('x7.shape',x.shape)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
#        print('x8.shape',x.shape)

        # 全连接层
        x = x.view(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x


def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 100
    predicted = []

    for i in range(len(X)//batch_size):
        s = i*batch_size
        e = i*batch_size+batch_size

        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    predicted = predicted.data.cpu().numpy()
    """
    设置评估指标：
    acc：准确率
    auc:AUC 即 ROC 曲线对应的面积
    recall:召回率
    precision:精确率
    fmeasure：F值
    """
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results


# 定义网络
net = EEGNet()
# 定义二分类交叉熵 (Binary Cross Entropy)
criterion = nn.BCELoss()
# 定义Adam优化器
optimizer = optim.Adam(net.parameters())

"""
生成训练数据集，数据集有100个样本
训练数据X_train:为[0,1)之间的随机数;
标签数据y_train:为0或1
"""
X_train = np.random.rand(100,1, 120, 64).astype('float32')
y_train = np.round(np.random.rand(100).astype('float32')) 
"""
生成验证数据集，数据集有100个样本
验证数据X_val:为[0,1)之间的随机数;
标签数据y_val:为0或1
"""
X_val = np.random.rand(100,1,120, 64).astype('float32')
y_val = np.round(np.random.rand(100).astype('float32'))
"""
生成测试数据集，数据集有100个样本
测试数据X_test:为[0,1)之间的随机数;
标签数据y_test:为0或1
"""
X_test = np.random.rand(100, 1,120, 64).astype('float32')
y_test = np.round(np.random.rand(100).astype('float32'))

batch_size = 32
# 训练 循环
for epoch in range(1): 
    print("\nEpoch ", epoch)

    running_loss = 0.0
    for i in range(len(X_train)//batch_size-1): #有几个batch
        s = i*batch_size 
        e = i*batch_size+batch_size

        inputs = torch.from_numpy(X_train[s:e])
        labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        print('inputs',inputs.shape)  #[]
        print('labels',labels.shape)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print('outputs.shape',outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # 验证
    params = ["acc", "auc", "fmeasure"]
    print(params)
    print("Training Loss ", running_loss)
    print("Train - ", evaluate(net, X_train, y_train, params))
#    print("Validation - ", evaluate(net, X_val, y_val, params))
    print("Test - ", evaluate(net, X_test, y_test, params))