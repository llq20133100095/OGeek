#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:25:28 2018

@author: llq
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

train_data = pd.read_table('./data/oppo_round1_train_20180929.txt', 
        names= ['prefix','query_prediction','title','tag','label'], header= None).astype(str)

val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt', 
        names = ['prefix','query_prediction','title','tag','label'], header = None).astype(str)
test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',
        names = ['prefix','query_prediction','title','tag'],header = None).astype(str)

train_data = train_data[train_data['label'] != '音乐' ]
test_data['label'] = -1

train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']

#单个特征
for item in items:
    #通过item对label进行分割
    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    #统计item的点击次数
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
#交叉特征
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')
#delete the column 
train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)


def onehot_feature(train_data_,test_data_):
    one_hot_feature=['prefix_count','prefix_click','title_click','title_count','tag_click',\
           'tag_count','prefix_title_click', 'prefix_titlecount','prefix_tag_click','prefix_tagcount',\
           'title_tag_click','title_tagcount'] 
    enc = OneHotEncoder()
    train_data_=train_data_.fillna("0")
    test_data_=test_data_.fillna("0")
    data=pd.concat([train_data_,test_data_])
    
    #unchange feature: sparse matrix
    train_x=train_data_[['prefix_ctr']]
    test_x=test_data_[['prefix_ctr']]
    unchange_feature=['title_ctr','tag_ctr','prefix_title_ctr','prefix_tag_ctr','title_tag_ctr']
    for i in unchange_feature:
        train_x=np.hstack((train_x,train_data_[[i]]))
        test_x=np.hstack((test_x,test_data_[[i]]))
    train_x=train_x.astype(np.float)
    test_x=test_x.astype(np.float)
    
    #concate the one-hot encode
    for feature in one_hot_feature:
        data[feature]=data[feature].apply(lambda x:int(x))
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train_data_[feature].values.reshape(-1, 1))
        test_a = enc.transform(test_data_[feature].values.reshape(-1, 1))
        train_x=sparse.hstack((train_x,train_a))
        test_x=sparse.hstack((test_x,test_a))
    print('one-hot prepared !')
    
    return train_x,test_x
    

print('train beginning')

#X = np.array(train_data_.drop(['label'], axis = 1))
#y = np.array(train_data_['label'])
#X_test_ = np.array(test_data_.drop(['label'], axis = 1))
X,X_test_=onehot_feature(train_data_,test_data_)
X=sparse.lil_matrix(X)   #this can use index
y = np.array(train_data_['label'])
    
print('================================')
print(X.shape)
print(y.shape)
print('================================')


xx_logloss = []
xx_submit = []
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',   #binary class
    'metric': 'binary_logloss',  #度量
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,    #select 90% feature
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

best_f1_score=0
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,  #num_iterations
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    
    f1_sco=f1_score(y_test, np.where(gbm.predict(X_test, num_iteration=gbm.best_iteration)>0.4, 1,0))
    print(f1_sco)
    if(best_f1_score<f1_sco):
        best_f1_score=f1_sco
    xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
    #5th predict
    xx_submit.append(gbm.predict(X_test_, num_iteration=gbm.best_iteration))

print(best_f1_score)
print('train_logloss:', np.mean(xx_logloss))
s = 0
for i in xx_submit:
    s = s + i

#calculate the mean of 5th prediction
test_data_['label'] = list(s / N)
#round:返回浮点数x的四舍五入值
test_data_['label'] = test_data_['label'].apply(lambda x: round(x))
print('test_logloss:', np.mean(test_data_.label))
test_data_['label']=test_data_['label'].apply(lambda x:int(x))
test_data_['label'].to_csv('./submit/result.csv',index = False)