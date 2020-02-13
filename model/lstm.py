import torch
from torch import nn
import torch.nn.functional as F

#LSTM的块
class lstm_block(nn.Module):
    def __init__(self,input_size,time_step,hidden_size=128,num_layers=2):
        super(lstm_block, self).__init__()
        self.input_size=input_size #输入大小
        self.time_step=time_step #时间步长

        self.lstm = nn.LSTM(         
            input_size=input_size,
            hidden_size=hidden_size,        
            num_layers=num_layers,          
            batch_first=True,       
        )

    def forward(self, x):
        x=x.view(-1, self.time_step, self.input_size)
        r_out, (h_n, h_c) = self.lstm(x, None)  
        x=r_out[:, -1, :]
        return x

class lstm(nn.Module):
    def __init__(self,input_size,time_step,num_classes,hidden_size=128,num_layers=2):
        super(lstm, self).__init__()
        self.input_size=input_size #输入大小
        self.time_step=time_step #时间步长
        self.point = input_size*time_step #点数

        self.lstm1 = lstm_block(input_size, time_step)
        self.lstm2 = lstm_block(input_size, time_step)
        self.lstm3 = lstm_block(input_size, time_step)
        self.lstm4 = lstm_block(input_size, time_step)
        self.lstm5 = lstm_block(input_size, time_step)
        self.fc = nn.Linear(hidden_size*5, num_classes)

    def forward(self, x):
        y = []
        for i in range(5):
            y.append(x[:,self.point*i:self.point*(i+1)].view(-1, self.time_step, self.input_size))
        y1 = self.lstm1(y[0])
        y2 = self.lstm2(y[1])
        y3 = self.lstm3(y[2])
        y4 = self.lstm4(y[3])
        y5 = self.lstm5(y[4])
        x = torch.cat((y1,y2,y3,y4,y5), 1) 
        x = self.fc(x)
        return x