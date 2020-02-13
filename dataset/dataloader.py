from torchvision import transforms, datasets
import h5py
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

#参考：https://blog.csdn.net/shwan_ma/article/details/100012808
#https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/blob/master/dataset.py

class DataFromMat(Dataset):
    def __init__(self, filepath, training_test , standardize=True):

        electrodes = 22  #22路脑电电极
        X, y = [], []
        #------------------加载所有的.mat数据------------------
        for i in range(9):
            A01T = h5py.File(filepath +'A0'+ str(i + 1) + 'T_slice.mat', 'r')
            X1 = np.copy(A01T['image']) 
            X1 = X1[:, :electrodes, :]
            X.append(np.asarray(X1,dtype=np.float32))
            
            y1 = np.copy(A01T['type']) 
            y1 = y1[0, 0:X1.shape[0]:1] #每个对象每次试验的标签
            y.append(np.asarray(y1, dtype=np.int32))
        
        #-----------------------删除受试对象中存在空值的某次实验-------------------------
        for subject in range(9):
            delete_list = [] #删除列表，删除存在空值的某次实验
            for trial in range(288):
                if np.isnan(X[subject][trial, :, :]).sum() > 0:
                    delete_list.append(trial) 
#                   print('delete_list',delete_list)
            X[subject] = np.delete(X[subject], delete_list, 0)
            y[subject] = np.delete(y[subject], delete_list)
            y = [y[i] - np.min(y[i]) for i in range(len(y))] #9个对象的标签，转换成0，1,2,3
            
        #把所有人的脑电信号都放在一起
        signals_all = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8])) #信号
        labels_all = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8])) #标签
#        print('signals_all.shape',signals_all.shape)
#        print('labels_all.shape',labels_all.shape)
        
        last_training_index = int(signals_all.shape[0]*0.8)
        
        #--------------按照0.8/0.2的比例划分训练/测试---------------
        if  training_test == 'train':
            self.data = torch.tensor(signals_all[:last_training_index, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[:last_training_index]) 
        
        elif training_test == 'test':
            self.data = torch.tensor(signals_all[last_training_index:, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[last_training_index:])
        
        #如果是标准化的，则减去均值，并除以方差
        if standardize:
            data_mean = self.data.mean(0) 
            data_var = np.sqrt(self.data.var(0))
            self.data = (self.data -data_mean)/data_var
              
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return  data,label
    
    def __len__(self):
        return self.data.shape[0]
    
    
def get_data(filepath, standardize=True):
    train_dataset = DataFromMat(filepath, 'train')
    test_dataset = DataFromMat(filepath, 'test')
    train_loaders = DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=4)
    test_loaders = DataLoader(test_dataset, batch_size=64,shuffle=True, num_workers=4)
    train_sizes = len(train_dataset)   
    test_sizes = len(test_dataset)                      
    return train_loaders, test_loaders,train_sizes,test_sizes


if __name__ == '__main__':
    filepath = "./data/"
    #将一个的部分数据作为测试集
    train_loader,test_loader = get_data(filepath)
    
    for signals, labels in test_loader:
        print('signals.shape',signals.shape)
        print('labels.shape',labels.shape)
        
    

