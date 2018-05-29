# -*- coding: utf-8 -*-
"""
Created on 2018.05.01

@author: jaylee
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

#%% 交易矩阵参数设定
read_path = "数据协整\\"
result_path = "回测结果\\"
print("=========读取数据=========")
# 收盘价
close=pd.read_excel(read_path+'close.xlsx')
# 流通市值
mkt_cap_float= pd.read_excel(read_path+"mkt_cap_float.xlsx")
# 交易信息
ipo_listdays=pd.read_excel(read_path+'ipo_listdays.xlsx')
trade_status=pd.read_excel(read_path+'trade_status.xlsx')
un_st_flag=pd.read_excel(read_path+'un_st_flag.xlsx')
# 中信一级行业分类
industry_citic = pd.read_excel(read_path+"industry_d1.xlsx")
sec_name = pd.read_excel(read_path+"sec_name.xlsx")
# 时间和股票标签
timelist=list(close.columns)
stocklist=list(close.index)

#======选股设置
# 可交易信号
trade_status_flag=deepcopy(trade_status)
trade_status_flag[(trade_status_flag=='交易') | (trade_status_flag=='停牌1小时')]=1
trade_status_flag[trade_status_flag!=1]=np.nan
trade_status_flag=trade_status_flag.ix[:,range(0,len(trade_status_flag.T))]

# 排除新股：上市一个月以内的股票去除
ipo_flag2=deepcopy(ipo_listdays)
ipo_flag2[ipo_listdays<=30]=np.nan
ipo_flag2[ipo_listdays>30]=1 

## 市值过滤
dnbound=0
upbound=1
mkt_flag=deepcopy(mkt_cap_float)
mkt_flag[:]=np.nan
quantiles=mkt_cap_float[mkt_cap_float.notnull()].quantile([dnbound,upbound])
mkt_flag[(mkt_cap_float>quantiles.loc[dnbound,:]) & (mkt_cap_float<quantiles.loc[upbound,:])]=1 

# 如需去除某些行业，在此处进行处理
industry_d1_flag=deepcopy(industry_citic)
industry_d1_flag[industry_d1_flag.isnull()]=np.nan
industry_d1_flag[industry_d1_flag.notnull()]=1

# 能否交易的矩阵
can_trade_mat=np.array(trade_status_flag,dtype="float64")*np.array(un_st_flag,dtype="float64")*\
                np.array(ipo_flag2,dtype="float64")*np.array(industry_d1_flag,dtype="float64")

can_trade_df=pd.DataFrame(can_trade_mat,index=stocklist,columns=timelist) 

industry_d1 = industry_citic

#%% 各种指标计算函数
    ##绩效回测
def Calcuate_performance_indicators(dataframe,period):
    ## 年化收益率
    Total_return=(dataframe+1).cumprod(axis=1).ix[:,-1]-1
    Annualized_Return = (dataframe+1).cumprod(axis=1).ix[:,-1]**(12/len(dataframe.columns[:-1]))-1;
    Annualized_Volatility = dataframe.std(axis=1)*period**0.5;
    Sharp = Annualized_Return/Annualized_Volatility;
    Maxdrawndown=[];
    for i in range(len(dataframe)):
        l=[];
        for j in range(len(dataframe.columns)):
            l.append(((dataframe+1).cumprod(axis=1).ix[i,j]-(dataframe+1).cumprod(axis=1).ix[i,:j].max())/(dataframe+1).cumprod(axis=1).ix[i,:j].max())
        Maxdrawndown.append(np.nanmin(np.array(l)));

    l1=[Total_return.ix[0],Annualized_Return.ix[0],Annualized_Volatility.ix[0],Sharp.ix[0],Maxdrawndown[0]]
    df=pd.DataFrame()
    df=df.reindex(index=['总收益率','年化收益率','年化波动率','夏普率','最大回撤'],columns=['回测数据'])
    df.ix[:,0]=l1
    return(df)

def Cal_returns(trade_df,name='组合'):
    # 计算市值均值
    ave_mkt=pd.DataFrame((trade_df*mkt_cap_float).mean(),columns=[name])
    
    # 计算市值加权
    mkt=deepcopy(mkt_cap_float)
    mkt=mkt[trade_df==1]
    mkt_ratio=mkt/mkt.sum()
    
    # 采取等权重 (若需采取市值加权，则将此段注释即可)
    mkt_ratio=trade_df/trade_df.count()
    
    # 计算收益率
    f = trade_df
    close1 = np.array(f.ix[:,:-1])*np.array(close.ix[:,:-1])
    close2 = np.array(f.ix[:,:-1])*np.array(close.ix[:,1:])
    net = (close2-close1)/close1

    return_net = pd.DataFrame(index=['return'],columns=f.columns)
    return_net.ix[:,0] = 0
    return_net.ix[:,1:] = np.nansum(net*np.array(mkt_ratio.ix[:,:-1]),axis=0)

    # 记录收益
    returns=return_net.T
    returns.columns=[name]
    # 计算累积收益率
    cum_netvalue=(return_net+1).cumprod(axis=1).T
    cum_netvalue.columns=[name]
    
    # 计算多种指标
    indi=Calcuate_performance_indicators(return_net,12)
    indi.columns=[name]
    
    # 计算仓位信息
    tradenum=pd.DataFrame(trade_df.sum(),columns=[name])
    return net,tradenum,cum_netvalue,indi,ave_mkt,returns

name=None
# 计算回测绩效
def Cal_flag(flag_df,name='组合'):
    net,tradenum,cum_netvalue,indi,ave_mkt,returns=Cal_returns(flag_df,name=name)   
    # 记录信息
    record3={}
    record3['tradenum']=tradenum
    record3['netvalues']=cum_netvalue
    record3['ave_mkt']=ave_mkt/100000000
    record3['returns']=returns
    record3['trade_df']=flag_df
    record3['performances']=indi
    return record3

#返回收全A益率矩阵
return_mat=Cal_returns(can_trade_df)[0]
return_matrix=pd.DataFrame(return_mat, index=stocklist, columns=timelist[1:])
return_matrix.to_excel('return.xlsx')
can_trade_df.to_excel('can_trade_df.xlsx')
#%%
# 计算指数
def run_index(index, startime, endtime, name='指数'):
    a=(np.array(index.ix[:,1:])-np.array(index.ix[:,:-1]))/np.array(index.ix[:,:-1])
    df_a = pd.DataFrame(a,index=[name], columns=timelist[1:])
    base_return=pd.DataFrame(index=[name],columns=timelist)
    base_return.ix[:,startime:] = df_a.ix[:,startime:]
    base_return.ix[:,:startime] = 0
    base_cumreturn=(base_return+1).cumprod(axis=1).T
    indi=Calcuate_performance_indicators(base_return,12)
    indi.columns=[name]
    
    # 记录收益
    returns=base_return.T
    returns.columns=[name]
    
    # 记录信息
    record={}
    record['cum_netvalue']=base_cumreturn
    record['indi']=indi
    record['returns']=returns
    return record

# 计算超额收益
def run_hedge(return_df, model_name, hedge='沪深300'):
    ls_returns=pd.DataFrame(return_df[model_name]-return_df[hedge],columns=['超额收益'])
    ls_return_net=deepcopy(ls_returns)
    ls_return_net.columns=['return']
    ls_return_net=ls_return_net.T
    ls_performance=Calcuate_performance_indicators(ls_return_net,12)
    ls_performance.columns=['对冲'+hedge+'组合绩效']
    ls_returns['对冲'+hedge+'累积收益率']=(ls_returns['超额收益']+1).cumprod(axis=0)-1
    return ls_returns,ls_performance

# 股票池
def create_stocktable(stockpool,j=-1):
    stock_table=pd.DataFrame(columns=[['证券代码','持仓权重','调整日期']])
    l=stockpool.ix[:,j]
    a=list(l[l==1].index)
    if len(a)==0:
       stock_table['证券代码']=a
       stock_table['调整日期']=stockpool.columns[j]
       stock_table['持仓权重']=1.0
       return stock_table
    stock_table['证券代码']=a
    stock_table['调整日期']=stockpool.columns[j]
    stock_table['持仓权重']=1.0/len(a)
    return stock_table

def annual_retrn(return_df):
    year_start=['2006-12-15','2007-12-15','2008-12-15','2009-12-15',
            '2010-12-15','2011-12-15','2012-12-15','2013-12-15','2014-12-15',
            '2015-12-15','2016-12-15']
    year_end=['2008-01-15','2009-01-15','2010-01-15',
            '2011-01-15','2012-01-15','2013-01-15','2014-01-15','2015-01-15',
            '2016-01-15','2017-01-15','2018-01-15']
    year_name=['2007','2008','2009','2010',
            '2011','2012','2013','2014','2015',
            '2016','2017']
    year_return_df=pd.DataFrame()
    for year in range(len(year_start)):
        syear=year_start[year]
        eyear=year_end[year]
        r=return_df
        sub_return_net=r.ix[(r.index>syear) & (r.index<eyear),:]
        sub_return_net=sub_return_net.ix[1:,:]
        total_return=(sub_return_net.T+1).cumprod(axis=1).ix[:,-1]-1
        total_return=pd.DataFrame(total_return)
        total_return.columns=[year_name[year]+'年收益绩效']
        year_return_df=year_return_df.join(total_return,how='outer')
    year_return_df=year_return_df.T
    year_return_df['最优组合']=year_return_df.idxmax(axis=1)
    return year_return_df
    
#%% 统计入选股票池的各种信息
def statistics_information(trade_df, startime, endtime, model_name):
    
    # 持仓统计信息(行业入选次数和个股入选次数)
    a = industry_citic[trade_df == 1]
    b = pd.value_counts(a.values.ravel())
    industryholding = pd.DataFrame(b, columns=['行业入选次数'])
    industryholding['行业占比'] = industryholding['行业入选次数'] / industryholding['行业入选次数'].sum()

    a=trade_df.sum(axis=1)
    stockholding=pd.DataFrame(a,columns=['入选次数'])
    stockholding['所属行业']=industry_citic.ix[:,-1]
    stockholding['股票名称']=sec_name
    stockholding=stockholding.sort_values('入选次数',ascending=False)

    # 市值统计
    mean_mkt = pd.DataFrame((trade_df * mkt_cap_float).mean(), columns=[model_name])
    mean_mkt['A股市值均值'] = (mkt_cap_float).mean()
    mean_mkt = mean_mkt / 100000000

    median_mkt = pd.DataFrame((trade_df * mkt_cap_float).median(), columns=[model_name])
    median_mkt['A股市值中位数'] = (mkt_cap_float).median()
    median_mkt = median_mkt / 100000000

    # 计算换手率
    mat1 = np.array(trade_df.ix[:, 1:])
    mat2 = np.array(trade_df.ix[:, :-1])
    turnover_mat = 1 - np.nansum(mat1 * mat2, axis=0) / np.nansum(mat2, axis=0)
    turnover_df = pd.DataFrame(index=[model_name], columns=trade_df.columns[1:])
    turnover_df.ix[:, :] = turnover_mat
    turnover_df = turnover_df.T

    hs300=pd.read_excel(read_path+"hs300.xlsx")
    zz500=pd.read_excel(read_path+"zz500.xlsx")
    
    record0 = Cal_flag(trade_df, model_name)
    record1 = run_index(hs300, startime, endtime, name='沪深300')
    record2 = run_index(zz500, startime, endtime, name='中证500')
    recordList = [record1, record2]
    netvalue_df = record0["netvalues"]
    preformance_df = record0["performances"]
    return_df = record0["returns"]
    for df in recordList:
        netvalue_df = netvalue_df.join(df['cum_netvalue'], how='outer')
        preformance_df = preformance_df.join(df['indi'], how='outer')
        return_df = return_df.join(df['returns'], how='outer')

    excess_hs300, p1 = run_hedge(return_df, model_name, hedge='沪深300')
    excess_zz500, p2 = run_hedge(return_df, model_name, hedge='中证500')
    
    #年度收益比较
#    year_return_df  = annual_retrn(return_df)
    
    #记录所有结果
    writer1 = pd.ExcelWriter(result_path + model_name + '.xlsx', engine='xlsxwriter',
                             datetime_format='yyyy/mm/dd')

    workbook = writer1.book

    format1 = workbook.add_format({'num_format': '0.00'})
    format2 = workbook.add_format({'num_format': '0.00%'})

    netvalue_df.to_excel(writer1, model_name+'净值曲线')
    preformance_df.T.to_excel(writer1, model_name + '回测绩效')
    record0["tradenum"].to_excel(writer1, '持仓数量')

#    year_return_df.to_excel(writer1,'年度收益绩效')
    excess_hs300.to_excel(writer1, '超额收益-HS300')
    excess_zz500.to_excel(writer1, '超额收益-ZZ500')
    industryholding.to_excel(writer1, '持仓分析-行业')
    stockholding.to_excel(writer1, '持仓分析-个股')
    turnover_df.to_excel(writer1, '换手率')

    mean_mkt.to_excel(writer1, '市值均值')
    median_mkt.to_excel(writer1, '市值中位数')

    worksheet = writer1.sheets[model_name+'回测绩效']
    worksheet.set_column('B:D', 8, format2)
    worksheet.set_column('E:E', 8, format1)
    worksheet.set_column('F:F', 8, format2)

    writer1.save()