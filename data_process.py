# -*- coding: utf-8 -*-
"""
Created on 2018.05.05

@author: jaylee
"""
import torch
import pandas as pd
import back_test
import os
from copy import deepcopy
import torch.utils.data as Data
from sklearn.preprocessing import Imputer

#%%
#获取./data文件夹下所有因子数据
def get_factors_data():
    factor_list=[]
    file_list = [file for file in os.listdir('./data')]
    factor_name = [os.path.splitext(file)[0] for file in os.listdir('.\data')]
    for file in file_list:
        df = pd.read_excel("./data/" + file)
        df.columns = timelist#协整因子数据
        factor_list.append(df)
    return factor_list, factor_name

#%%构造符合Torch形式的训练数据集
def get_train_data(cur_time, period):
    start_index = timelist.index(pd.to_datetime(cur_time)) - period  
    end_index = timelist.index(pd.to_datetime(cur_time)) 
    end_time = timelist[end_index]
    pre_timelist = timelist[start_index:end_index]
#    print(pre_timelist)
#    print(end_time)
    can_trade_stock = can_trade_df.ix[:, end_time].to_frame().dropna().index
    return_ret = return_matrix.ix[can_trade_stock, end_time].sort_values(ascending=False).to_frame().dropna()
    #取收益率前30%与后30%数据，二值化置为0和1
    se1 = deepcopy(return_ret.ix[:int(len(return_ret)*3/10), :])
#    se1.ix[:, :] = 1.0
    se2 = deepcopy(return_ret.ix[int(len(return_ret)*7/10):, :])
#    se2.ix[:, :] = 0.0
    return_processed = se1.append(se2)
    #可交易股票池
    trade_stocks = return_processed.index
    factor_data_list = []
    for i in range(len(trade_stocks)):
        #记录每一支股票的因子数据
        factor_data = pd.DataFrame(index=pre_timelist, columns=factor_name)
        for j in range(len(factor_list)):
            factor_data.ix[:,j] = factor_list[j].ix[trade_stocks[i],pre_timelist]
        #若某列因子数据全为空，则剔除该股票
        df = deepcopy(factor_data)
        df = df.dropna(axis=1, how='all')
        if df.shape[1] != factor_data.shape[1]:           
            return_processed = return_processed.drop(trade_stocks[i])
            continue
        #以特征列的均值处理空值
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(factor_data.values)
        factor_data_value = imp.transform(factor_data.values)
        factor_data_list.append(factor_data_value)
    #构造tensor形式的训练数据    
    train_data = torch.zeros(len(factor_data_list), period, len(factor_list))
    for i in range(len(factor_data_list)):
        train_data[i] = torch.from_numpy(factor_data_list[i]).float()
    #构造tensor形式的标签数据
    train_data_label = torch.from_numpy(return_processed.values).float()
    return train_data, train_data_label

def get_tarin_data_series(cur_time, period, N_month):
    cur_index = timelist.index(pd.to_datetime(cur_time))
    train_data_timelist=timelist[cur_index-N_month:cur_index]
#    print(train_data_timelist)
#    print(train_data_timelist[0])
    tenor_data0, tenor_label0 = get_train_data(train_data_timelist[0], period)
    for i in range(len(train_data_timelist)-1):
#        print(train_data_timelist[i+1])
        tenor_data, tenor_label = get_train_data(train_data_timelist[i+1], period)
        tenor_data0 = torch.cat((tenor_data0, tenor_data), 0)
        tenor_label0 = torch.cat((tenor_label0, tenor_label), 0)
    torch_dataset = Data.TensorDataset(tenor_data0, tenor_label0)
    return  torch_dataset

#%%构造符合Torch形式的训练数据集
def get_train_dataset(cur_time, period):
    start_index = timelist.index(pd.to_datetime(cur_time)) - 2*period - 1
    end_index = timelist.index(pd.to_datetime(cur_time)) - period - 1
    end_time = timelist[end_index]
    pre_timelist = timelist[start_index:end_index]
    can_trade_stock = can_trade_df.ix[:, end_time].to_frame().dropna().index
    return_ret = return_matrix.ix[can_trade_stock, end_time].sort_values(ascending=False).to_frame().dropna()
    #取收益率前30%与后30%数据，二值化置为0和1
    se1 = deepcopy(return_ret.ix[:int(len(return_ret)*3/10), :])
#    se1.ix[:, :] = 1.0
    se2 = deepcopy(return_ret.ix[int(len(return_ret)*7/10):, :])
#    se2.ix[:, :] = 0.0
    return_processed = se1.append(se2)
    #可交易股票池
    trade_stocks = return_processed.index
    factor_data_list = []
    for i in range(len(trade_stocks)):
        #记录每一支股票的因子数据
        factor_data = pd.DataFrame(index=pre_timelist, columns=factor_name)
        for j in range(len(factor_list)):
            factor_data.ix[:,j] = factor_list[j].ix[trade_stocks[i],pre_timelist]
        #若某列因子数据全为空，则剔除该股票
        df = deepcopy(factor_data)
        df = df.dropna(axis=1, how='all')
        if df.shape[1] != factor_data.shape[1]:           
            return_processed = return_processed.drop(trade_stocks[i])
            continue
        #以特征列的均值处理空值
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(factor_data.values)
        factor_data_value = imp.transform(factor_data.values)
        factor_data_list.append(factor_data_value)
    #构造tensor形式的训练数据    
    train_data = torch.zeros(len(factor_data_list), period, len(factor_list))
    for i in range(len(factor_data_list)):
        train_data[i] = torch.from_numpy(factor_data_list[i]).float()
    #构造tensor形式的标签数据
    train_data_label = torch.from_numpy(return_processed.values).float()
    #前period时段的因子值和收益率序列
    torch_dataset = Data.TensorDataset(train_data, train_data_label)
    return torch_dataset

#%%构造符合Torch形式的测试数据集
def get_test_data(cur_time, period):
    start_index = timelist.index(pd.to_datetime(cur_time))-period
    end_index = timelist.index(pd.to_datetime(cur_time))
    pre_timelist = timelist[start_index:end_index]
    can_trade_stock = can_trade_df.ix[:, cur_time].to_frame().dropna().index
    return_ret = return_matrix.ix[can_trade_stock, cur_time].sort_values(ascending=False).to_frame().dropna()
    trade_stocks = return_ret.index
    factor_data_list = []
    for i in range(len(trade_stocks)):  
        factor_data = pd.DataFrame(index=pre_timelist, columns=factor_name)
        for j in range(len(factor_list)):
            factor_data.ix[:,j] = factor_list[j].ix[trade_stocks[i],pre_timelist]
        #若某列因子数据全为空，则剔除该股票
        df = deepcopy(factor_data)
        df = df.dropna(axis=1, how='all')
        if df.shape[1] != factor_data.shape[1]:
            return_ret = return_ret.drop(trade_stocks[i])
            continue
        #以特征列的均值处理空值
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(factor_data.values)
        factor_data_value = imp.transform(factor_data.values)
        factor_data_list.append(factor_data_value)
    #构造tensor形式的测试数据   
    test_data = torch.zeros(len(factor_data_list), period, len(factor_list))    
    for i in range(len(factor_data_list)):
        test_data[i] = torch.from_numpy(factor_data_list[i]).float()
    return return_ret, test_data

#%%测试代码
can_trade_df = back_test.can_trade_df
return_matrix = back_test.return_matrix
timelist = back_test.timelist
stocklist = back_test.stocklist
factor_list, factor_name = get_factors_data()

'''Test code'''
if __name__ == "__main__":
    period = 5
    N_month = 5
    cur_time = "2010-01-29"
    torch_dataset = get_tarin_data_series(cur_time, period, N_month)
#    trade_ret, test_data = get_test_data(cur_time, period)
#    print(len(trade_ret))
#    print(test_data.shape)
#    train_dataset = get_train_dataset(cur_time, period)


