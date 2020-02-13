#参考：https://github.com/tom-beer/deep-sleep-mind/blob/master/networks.py
 #https://github.com/HypoX64/candock/blob/master/models/cnn_1d.py
import argparse
import warnings
warnings.filterwarnings('ignore')
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import logging
import time
from torch.autograd import Variable

#from preprocessing import *
from dataset.dataloader import get_data
from model.cnn_1d import ConvNet, OneChannelConv,  BaseClassifierNet ,MultiChannelConv
from model.resnet_1d import resnet18
from model.multi_scale_resnet_1d import Multi_Scale_ResNet
from model.graphnet import GCN3

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import roc_auc_score
from scipy import interp

#logging 将日志同时输出到屏幕和日志文件
logger=logging.getLogger()
logger.setLevel(logging.INFO) #log the info

handler = logging.FileHandler("C:/Users/tworld/Desktop/Code/wll/EEG/eeg_wll/cnn_gcn/log/test.log")
handler.setLevel(logging.INFO)

console=logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler) #log on txt
logger.addHandler(console) #log on console


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_classes', type=int, default=4,
                    help='number of categories')
parser.add_argument('--k_neighbor', type=int, default=4,
                    help='number of neighbor')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model,trainloader, testloader, trainsize, criterion, optimizer,scheduler, num_epochs):
#def train_model(model,X_train,y_train,X_test,y_test,criterion, optimizer, num_epochs):
    since = time.time()
    best_acc = 0.0
    best_epoch=0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        running_loss = 0.0
        running_corrects = 0
        
        #train部分
        for inputs, labels in trainloader:
            
            # wrap them in Variable 将它们转变为变量
            inputs, labels = Variable(inputs), Variable(labels)
            labels = torch.squeeze(labels).long()
#            print('input.shape',inputs.shape) #[64,22,1000]
            
#            print('labels',labels)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs =model(inputs)
#            print('outputs',outputs.shape)
            _, preds = torch.max(outputs, 1)
#            print('preds',preds)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
       
        scheduler.step()
        
        epoch_loss = running_loss / trainsize
        epoch_acc = running_corrects.double() / trainsize
        
        # 每个epoch输出一次准确率
#        print("Training Loss ", epoch_loss)
#        print("Training acc ", epoch_acc)
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        #测试部分
        params = ["acc", "precision", "recall","fmeasure"]
        test_result = evaluate(model, testloader, params)
        test_acc = test_result[0]
        test_precision = test_result[1]
        test_recall = test_result[2]
        test_f1 = test_result[3]
        logging.info('Test Acc: {:.4f}  precision: {:.4f} recall: {:.4f} F1: {:.4f}'.
                     format(test_acc, test_precision, test_recall, test_f1))

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch =epoch
            
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best test Acc : {:4f} on epoch {}'.format(best_acc,best_epoch))


def evaluate(model, testloader, params = ["acc"]):
    results = []
    y_labels =[]
    y_pred = []
    
    model.eval()  # Set models to evaluate mode

    for inputs, labels in testloader:
        
        inputs, labels = Variable(inputs), Variable(labels)
        labels = torch.squeeze(labels).long()
        
        outputs =model(inputs)
        _,pred = torch.max(outputs, 1)
#        print('pred',pred)
        
        #----------add-------
        for label in labels:
            y_labels.append(label)
                    
        pred_list =pred.cpu().tolist()
        for pred in pred_list:
            y_pred.append(pred)

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(y_labels, y_pred))
        if param == "auc":
            results.append(roc_auc_score(y_labels, y_pred,average='macro'))
        if param == "recall":
            results.append(recall_score(y_labels, y_pred,average='macro'))
        if param == "precision":
            results.append(precision_score(y_labels, y_pred,average='macro'))
        if param == "fmeasure":
            results.append(f1_score(y_labels,y_pred,average='macro'))
    return results


def train_model2(featuremodel,classmodel,trainloader, trainsize, criterion, optimizer,scheduler, num_epochs):
    since = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        running_loss = 0.0
        running_corrects = 0
        
        #train部分
        for inputs, labels in trainloader:
            
            batch_feature = torch.zeros(inputs.size(0),1)
            
            # wrap them in Variable 将它们转变为变量
            inputs, labels = Variable(inputs), Variable(labels)
            labels = torch.squeeze(labels).long()
            print('input.shape',inputs.shape) #(64,22,1000)
            for elecnode in range(inputs.size(1)):
                eachNode_input = torch.unsqueeze(inputs[:,elecnode,:], 1) #每一个节点通道的原始数据
#                print('eachNode_input.shape',eachNode_input.shape) #(64,1,1000)
                eachNode_feature = featuremodel(eachNode_input) #每一个节点通道的特征
#                print('eachNode_feature.shape',eachNode_feature.shape) #(64,16)
                batch_feature = torch.cat([batch_feature,eachNode_feature],dim=1)  #(64,352)   

#            print('batch_feature.shape',batch_feature.shape)
#            print('labels',labels)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs =classmodel(batch_feature)
#            print('outputs',outputs.shape)
            _, preds = torch.max(outputs, 1)
#            print('preds',preds)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
       
        scheduler.step()
        
        epoch_loss = running_loss / trainsize
        epoch_acc = running_corrects.double() / trainsize
    
        # 每个epoch输出一次准确率
#        print("Training Loss ", epoch_loss)
#        print("Training acc ", epoch_acc)
        logging.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
            

if __name__ == '__main__':
    
    #加载数据
#    datadir='C:/Users/tworld/Desktop/Code/wll/EEG/demo/Motor-Imagery-Tasks-Classification-using-EEG-data-master/data/'
    datadir = "./data/"
    train_loader,test_loader,train_sizes,test_sizes = get_data(datadir)
    
    
#    model_ft =ConvNet(inchannel=22, num_classes=args.num_classes)
#    model_ft =MultiChannelConv(inchannel=1, num_classes=args.num_classes,elecnodes=22)
#    model_ft =resnet18( num_classes=args.num_classes, pretrained=False)
#    model_ft =Multi_Scale_ResNet(inchannel=22, num_classes=args.num_classes)
    model_ft =GCN3( nclass=args.num_classes, k_neighbor=args.k_neighbor)
    model_ft = model_ft.to(device)
    print(model_ft)
    
    
#    feature_model = OneChannelConv(inchannel=1)
#    class_model = BaseClassifierNet(num_classes=4, feature_len=353)
    
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
#    optimizer_ft = optim.SGD(class_model.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model_ft = train_model(model_ft,train_loader,test_loader,train_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=args.epochs)
    
#    model_ft = train_model2(feature_model,class_model,test_loader, test_sizes, criterion, optimizer_ft,exp_lr_scheduler, num_epochs=args.epochs)
    

