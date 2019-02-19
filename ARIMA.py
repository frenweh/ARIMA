# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:08:39 2018

@author: wangyu
"""
#参数估计
import warnings  #忽略警告信息
import itertools
import pandas as pd
import numpy as np
import statsmodels #时间序列
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore") 
#原数据可视化
PARK = pd.read_csv('PARK.csv',index_col = 'Date',parse_dates=['Date'])

PARK.index = pd.to_datetime(PARK.index)
sub = PARK['2014-01':'2017-12']['PARK']
train = sub.ix['2014-01':'2017-01']
train_diff=train.diff(1).dropna()
test = sub.ix['2017-02':'2017-12']
test_diff=test.diff(1).dropna()
plt.figure(figsize=(10,10))

print('训练数据:')
print(train)
train.plot(color='blue',title='observed train data')
plt.plot(train)
plt.savefig('原始数据.png', dpi=128)  
plt.show()

#原始数据差分
PARK['PARK_diff_1'] = PARK['PARK'].diff(1)
PARK['PARK_diff_2'] = PARK['PARK_diff_1'].diff(1)
fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(131)
ax1.plot(PARK['PARK'])
plt.title('observed data')
ax2 = fig.add_subplot(132)
ax2.plot(PARK['PARK_diff_1'])
plt.title('First order difference')
ax3 = fig.add_subplot(133)
ax3.plot(PARK['PARK_diff_2'])
plt.title('Second order difference')
print('Final model diagnostics:')
plt.savefig('Final model diagnostics.png', dpi=128)
plt.show()

#训练数据的拖尾和截尾情况
fig = plt.figure(figsize=(12,8))
 
ax1 = fig.add_subplot(211)
fig = plot_acf(train, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
 
ax2 = fig.add_subplot(212)
fig = plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.savefig('acf and pacf of train data.png', dpi=128)
plt.show()
#一阶差分后训练数据的集中显示
def tsplot(y, lags=None, title='', figsize=(14, 8)):
 
    plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
 
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    plot_acf(y, lags=lags, ax=acf_ax)
    plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
 
tsplot(train_diff, title='Consumer Sentiment', lags=36)
plt.savefig('一阶差分后训练数据的相关图表.png', dpi=128)
plt.show() 
'''
# Initialize a DataFrame to store the results,，以BIC准则   
# 用“网格搜索”来迭代地探索参数的不同组合,相对最优模型识别：将p、d、q的可能值进行遍历，一次带入模型求最小的aic
p_min = 0
d_min = 0
q_min = 0
p_max = 3
d_max = 2
q_max = 3
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

results_listA = []
results_listB = []
results_listC = []
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = sm.tsa.ARIMA(train, order=(p,d,q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                               )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic        
        print('ARIMA{}-AIC:{}- BIC:{}-HQIC:{}'.format((p,d,q),results.aic,results.bic,results.hqic))
        results_listA.append([(p,d,q),results.aic])
        results_listB.append([(p,d,q),results.bic])
        results_listC.append([(p,d,q),results.hqic])
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
results_listA = np.array(results_listA)
results_listB = np.array(results_listB)
results_listC = np.array(results_listC)
lowest_AIC = np.argmin(results_listA[:, 1])
lowest_BIC = np.argmin(results_listB[:, 1])
lowest_HQIC = np.argmin(results_listC[:, 1])
print('---------------------------我是分割线-------------------------------')
print('ARIMA{} with lowest_AIC:{}'.format(
            results_listA[lowest_AIC, 0], results_listA[lowest_AIC, 1]))
print('ARIMA{} with lowest_BIC:{}'.format(
            results_listB[lowest_BIC, 0], results_listB[lowest_BIC, 1]))
print('ARIMA{} with lowest_HQIC:{}'.format(
            results_listB[lowest_HQIC, 0], results_listC[lowest_HQIC, 1]))
print('---------------------------我是分割线--------------------------------')
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.savefig('BIC heatmap.png', dpi=128)
plt.show()

#模型评价准则
train_results=sm.tsa.arma_order_select_ic(train_diff,max_ar=5,max_ma=5,ic=['aic','bic','hqic'],trend='nc')
print('AIC建议模型：', train_results.aic_min_order)
print('BIC建议模型：', train_results.bic_min_order)
print('HQIC建议模型：', train_results.hqic_min_order)
'''
#模型测试
model = sm.tsa.ARIMA(test, order=(2,0,1))#results_listB[lowest_BIC, 0])
results= model.fit(disp=-1)
print('Final model summary:')
print(results.summary().tables[1])
print('AIC:',results.aic,'BIC:', results.bic, 'HQIC:',results.hqic)
plt.figure(facecolor='white',figsize=(10,10))
results.resid.plot(color='red', label='Predict')
test_diff=test.diff(1).dropna()
test_diff.plot(color='blue', label='Original')
plt.legend(loc='best') 
result=(results.fittedvalues-test_diff)**2 #残差平方
#result.dropna(Replace=True)
plt.title('ARIMA RSS: %.4f' % sum(result))
plt.savefig('模型训练.png', dpi=128)
plt.show()

#模型检验
resid = results.resid #赋值
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
plt.savefig('模型检验.png', dpi=128)
plt.show()

#D-W检验
print('Durbin-Watson:' , sm.stats.durbin_watson(resid.values))
#qq图检验
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.savefig('qq test figure')
#白噪声检验
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# forecast方法会自动进行差分还原，仅限于支持的1阶和2阶差分
forecast_n = 68 #预测未来68个天走势
forecast = results.forecast(forecast_n)
forecast = forecast[0]
print (forecast)
#将预测的数据和原来的数据绘制在一起，使用开源库arrow增加数据索引
import arrow
def get_date_range(start, limit, level='day',format='YYYY-MM-DD'):
    start = arrow.get(start, format)  
    result=(list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level, start,limit=limit))))
    dateparse2 = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
    return map(dateparse2, result)
 
# 预测从2014-09-30开始，也就是测试数据最后一个数据的后一个日期
new_index = get_date_range('2017-12-03', forecast_n)
forecast_ARIMA = pd.Series(forecast, copy=True, index=new_index)
print (forecast_ARIMA.head())
##绘图如下
plt.figure(figsize=(10,10))
plt.plot(sub,label='Original',color='blue')
plt.plot(forecast_ARIMA, label='Forcast',color='red')
plt.legend(loc='best')
plt.title('forecast')
plt.savefig('预测.png', dpi=128)
plt.show()
