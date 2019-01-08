#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:24:06 2018

@author: llq
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import time

def length_feature(data):
    '''
    对vector feature构造长度特征
    '''
    vec_feature = ['query_prediction']
    
    for co in vec_feature:
        value = []
        lis = list(data[co].values)
        for i in range(len(lis)):
            value.append(len(lis[i].split(',')))
        col_name = co+'_length'
        data[col_name]=value
    return data

def count_feature(train_data_,test_data_):
    label_feature=['query_prediction_length'] 
    enc = LabelEncoder()
    data=pd.concat([train_data_,test_data_])
    
    #concate the one-hot encode
    for feature in label_feature:
        data[feature]=data[feature].apply(lambda x:int(x))
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train_data_[feature].values.reshape(-1, 1))
        test_a = enc.transform(test_data_[feature].values.reshape(-1, 1))
        train_data_[feature]=train_a
        test_data_[feature]=test_a
    print('label prepared !')
    
    return train_data_,test_data_

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

#构造长度特征
train_data=length_feature(train_data)
test_data=length_feature(test_data)


for item in items:
    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    #normailze
    se=temp[item+'_count'].values.astype(float)
    semax = se.max()
    semin = se.min()
    temp[item+'_nor'] = ((se-se.min())/(se.max()-se.min())*100).astype(int)
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')

#3 feature across
item_g = [items[0], items[1], items[2]]
temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
train_data = pd.merge(train_data, temp, on=item_g, how='left')
test_data = pd.merge(test_data, temp, on=item_g, how='left')

train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag', 'prefix_count', 'title_count', 'tag_count'], axis = 1)
test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag', 'prefix_count', 'title_count', 'tag_count'], axis = 1)

print('train beginning')

X = np.array(train_data_.drop(['label'], axis = 1))
y = np.array(train_data_['label'])
X_test_ = np.array(test_data_.drop(['label'], axis = 1))
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
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
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
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    
    val_test=gbm.predict(X_test, num_iteration=gbm.best_iteration)
    f1_sco=f1_score(y_test, np.where(val_test>0.4, 1,0))
    
    print(f1_sco)
    if(best_f1_score<f1_sco):
        best_f1_score=f1_sco
        
    xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
    xx_submit.append(gbm.predict(X_test_, num_iteration=gbm.best_iteration))

    
print(best_f1_score)
print('train_logloss:', np.mean(xx_logloss))

s = 0
for i in xx_submit:
    s = s + i

test_data_['label'] = list(s / N)
test_data_['label'] = test_data_['label'].apply(lambda x: round(x))
print('test_logloss:', np.mean(test_data_.label))
test_data_['label']=test_data_['label'].apply(lambda x:int(x))
test_data_['label'].to_csv('./submit/result_baseline.csv',index = False)
