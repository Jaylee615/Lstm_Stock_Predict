# -*- coding: utf-8 -*-
"""
Created on 2018.05.01

@author: jaylee
"""
import time
time1 = time.clock()#记录程序运行时间

import torch
from torch import nn
from torch.autograd import Variable
import back_test
import data_process
import pandas as pd
import numpy as np
from copy import deepcopy

#%%超参数
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 500
TIME_STEP = 6          # rnn time step 
INPUT_SIZE = 4         # rnn input size / n_features
LR = 0.01              # learning rate
top_k = 50             #选择预测收益最高的20只股票   
train_pre_month=6  
                            
#%%回测参数设置
model_name = "LSTM选股模型"
startime = "2010-01-29"
endtime = "2017-12-29"
timelist = back_test.timelist
can_trade_df = back_test.can_trade_df
start_index = timelist.index(pd.to_datetime(startime))
end_index = timelist.index(pd.to_datetime(endtime))
backtest_timelist = timelist[start_index:end_index]
#backtest_timelist = [timelist[start_index]]

#%%
class RNN(nn.Module):
    def __init__(self):
        
        super(RNN, self).__init__()
        
        self.bn_input = nn.BatchNorm1d(TIME_STEP , momentum=0.5)
        
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size = INPUT_SIZE,
            hidden_size = 20,         # rnn hidden unit
            num_layers = 2,           # number of rnn layer
#            dropout = 0.2,
            batch_first = True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )    
        
#        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = self.bn_input(x)
        r_out,(h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        r_out = self.bn_input(r_out)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
#print(rnn)

#%%
#选取预测收益最高的num支股票进入股票池
def stock_select(stockpools,output,num):
    stock_index = output.squeeze().topk(num)[1]
    stock_list = []
    for index in stock_index:
        stock_list.append(stockpools[index])
    return stock_list

#将回测时段入选股票构造交易矩阵
def trade_stockpool(stocks_list):
    trade_df = deepcopy(can_trade_df)
    trade_df.ix[:,:]=np.nan
    for i in range(len(backtest_timelist)):
        trade_df.ix[stocks_list[i],backtest_timelist[i]] = 1
    return trade_df

#计算预测正确率
def Cal_accuracy(pre_ret,act_ret):
    predicted_ret=deepcopy(pre_ret.squeeze().detach().numpy())
    actual_ret=deepcopy(act_ret.squeeze().values)
    multiply =predicted_ret*actual_ret
    accuarcy=len(multiply[multiply>=0])/len(multiply)
    return accuarcy

#%%
#回测模型
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   
loss_func = nn.MSELoss()  
trade_stocks_list = []
accuracy_list = []
for cur_time in backtest_timelist:
    print("=========当前回测时段: %s========="%cur_time)
    train_data = data_process.get_tarin_data_series(cur_time, TIME_STEP, train_pre_month)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_ret, test_data = data_process.get_test_data(cur_time, TIME_STEP)
    test_x = Variable(test_data)
    for epoch in range(EPOCH):
#        print("迭代次数：%d"%(epoch+1))
        for step, (x, y) in enumerate(train_loader):
#            print("step: ", step+1)
            b_x = Variable(x.view(-1, 6, 4))               
            b_y = Variable(y)                               
            output = rnn(b_x)  
            loss = loss_func(output, b_y)
            optimizer.zero_grad()                          
            loss.backward()                                 
            optimizer.step()
#            print('train loss: %.4f' % loss.data[0])
    rnn.eval()
    test_output = rnn(test_x)
    rnn.train()
    cur_trade_stock = stock_select(test_ret.index, test_output, top_k)
    trade_stocks_list.append(cur_trade_stock)
    accuracy=Cal_accuracy(test_output,test_ret)
    print("accuracy = ",accuracy)
    accuracy_list.append(accuracy)  
    
#测试集准确率
accuracy_df=pd.DataFrame(accuracy_list, index=backtest_timelist, columns=['正确率'])
accuracy_df.to_excel('./回测结果/模型评估.xlsx')
#将入选股票池构造可交易矩阵
can_trade_matrix = trade_stockpool(trade_stocks_list)
#统计组合绩效
back_test.statistics_information(can_trade_matrix, startime, endtime, model_name)
time2 = time.clock()
print("总花费时间", time2 - time1)
