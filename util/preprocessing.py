import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

#--------------导入原始数据,把所有的数据和标签放到了一个列表当中------------
def import_data(datadir,every=False):
    if every:
        electrodes = 25  #电极数，包括22路脑电电极和3路眼电电极
    else:
        electrodes = 22  #22路脑电电极
    X, y = [], []
    for i in range(9):
        A01T = h5py.File(datadir +'A0'+ str(i + 1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image']) 
#        print('X1.shape',X1.shape) #(288,22,1000)
#        X.append(X1[:, :electrodes, :])
        #9个对象的288次实验的特征，根据电极通道数取22或者25个通道的数据
        X.append(np.asarray(X1[:, :electrodes, :],dtype=np.float32)) 
        y1 = np.copy(A01T['type']) 
        y1 = y1[0, 0:X1.shape[0]:1] #每个对象每次试验的标签
#        print('y1',y1)
        y.append(np.asarray(y1, dtype=np.int32))
        
    for subject in range(9):
        delete_list = [] #删除列表，删除存在空值的某次实验
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial) 
#                print('delete_list',delete_list)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))] #9个对象的标签，转换成0，1,2,3
#    print('X.shape',X[8].shape) #[288-len(delete_list),22,1000]
#    print('y.shape',y[8].shape) #(288-len(delete_list))
    #返回所有对象的特征和标签
    return X, y


#------------按照一个对象划分训练/测试-----------------
def train_test_subject(X, y, train_all=True, standardize=True):
    
    l = np.random.permutation(len(X[0])) #permutation返回一个新的打乱顺序的数组，已经将数据进行打乱，但数据和标签是对应的
    X_test = X[0][l[:50], :, :] #第1个对象的其中50次实验作为测试集
    y_test = y[0][l[:50]] 

    #如果训练所有，把第1个对象的其中50次实验数据作为测试集，其他数据及其他人的数据作为训练集
    if train_all:
        X_train = np.concatenate((X[0][l[50:], :, :], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_train = np.concatenate((y[0][l[50:]], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    #否则，把第1个对象的其中50次实验数据作为测试集，其他数据作为训练集
    else:
        X_train = X[0][l[50:], :, :]
        y_train = y[0][l[50:]]

    X_train_mean = X_train.mean(0)  
    X_train_var = np.sqrt(X_train.var(0))

    #如果是标准化的，则减去均值，并除以方差
    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var   #(sample，通道数，时间长度)

#    X_train = np.transpose(X_train, (0, 2, 1)) #（2508,1000,22）
#    X_test = np.transpose(X_test, (0, 2, 1)) #(50,1000,22),进行维度1和2的转换，得到3D的数据，（样本数，时间长度，通道数）
    
    #转成Tensor
#    X_train =torch.Tensor(X_train)
#    X_test =torch.Tensor(X_test)
#    y_train =torch.Tensor(y_train)
#    y_test =torch.Tensor(y_test)
    
#    print('X_train.shape',X_train.shape) #(sample,22,1000)
#    print('X_test.shape',X_test.shape)
#    print('y_train.shape',X_train.shape)
#    print('y_test.shape',X_test.shape)
    
    return X_train, X_test, y_train, y_test


#--------------训练全部的数据--------------------
def train_test_total(X, y, standardize=True):
    #先把所有的对象数据进行合并
    X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
    y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))
    
    #选出其中的50次实验作为测试集
    l = np.random.permutation(len(X_total))
    X_test = X_total[l[:50], :, :]
    y_test = y_total[l[:50]]
    X_train = X_total[l[50:], :, :]
    y_train = y_total[l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

#    X_train = np.transpose(X_train, (0, 2, 1))
#    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    
    datadir='C:/Users/tworld/Desktop/Code/wll/EEG/demo/Motor-Imagery-Tasks-Classification-using-EEG-data-master/data/'
    #加载数据
    X, y = import_data(datadir,every=False)
    #将一个的部分数据作为测试集
    X_train,X_test,y_train,y_test = train_test_subject(X, y,train_all=False)
    print('X_train.shape',X_train.shape) #(sample,22,1000)
    print('X_test.shape',X_test.shape)
    print('y_train.shape',X_train.shape)
    print('y_test.shape',X_test.shape)