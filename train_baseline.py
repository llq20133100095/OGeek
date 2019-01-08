#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:11:01 2018

@author: llq
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

#train_data = pd.read_table('./data/oppo_round1_train_20180929.txt', 
#        names= ['prefix','query_prediction','title','tag','label'], header= None).astype(str)
#val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt', 
#        names = ['prefix','query_prediction','title','tag','label'], header = None).astype(str)
#test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',
#        names = ['prefix','query_prediction','title','tag'],header = None).astype(str)
#train_data = train_data[train_data['label'] != '音乐' ]
#test_data['label'] = -1
#
#train_data = pd.concat([train_data,val_data])
#train_data['label'] = train_data['label'].apply(lambda x: int(x))
#test_data['label'] = test_data['label'].apply(lambda x: int(x))
#items = ['prefix', 'title', 'tag']
#
#for item in items:
#    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
#    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
#    train_data = pd.merge(train_data, temp, on=item, how='left')
#    test_data = pd.merge(test_data, temp, on=item, how='left')
#for i in range(len(items)):
#    for j in range(i+1, len(items)):
#        item_g = [items[i], items[j]]
#        temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
#        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
#        train_data = pd.merge(train_data, temp, on=item_g, how='left')
#        test_data = pd.merge(test_data, temp, on=item_g, how='left')
#
##3 feature across
#item_g = [items[0], items[1], items[2]]
#temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
#temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
#train_data = pd.merge(train_data, temp, on=item_g, how='left')
#test_data = pd.merge(test_data, temp, on=item_g, how='left')
#
#train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
#test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)

from dnn_data import get_data
_,_,_,test_data_=get_data()
#add new feature:embedding similarity
X_train=np.load('./data_word2vec_feature/train_vector.npz')
X_test=np.load('./data_word2vec_feature/test_vector.npz')

X=np.array(X_train['train_data'][:-50000])
y=np.reshape(np.array(X_train['y_train'][:-50000],dtype=np.float),(-1,))

X_val_=np.array(X_train['train_data'][-50000:])
y_val=np.reshape(np.array(X_train['y_train'][-50000:],dtype=np.float),(-1,))

X_test_=np.array(X_test['test_data'])
y_test=np.reshape(np.array(X_test['y_test'],dtype=np.float),(-1,))

feature_names=['prefix_title_sim', 'prefix_title_dot',
       'text0_title_sim', 'text0_title_dot',
       'text1_title_sim', 'text1_title_dot',
       'text2_title_sim', 'text2_title_dot', 
       'text3_title_sim', 'text3_title_dot',
       'text4_title_sim', 'text4_title_dot',
       'text5_title_sim', 'text5_title_dot',
       'text6_title_sim', 'text6_title_dot',
       'text7_title_sim', 'text7_title_dot',
       'text8_title_sim', 'text8_title_dot',
       'text9_title_sim', 'text9_title_dot',
       'prefix_cut_in_title', 'prefix_cut_in_start_title',
       'query_prediction_length','is_in_title',
       'leven_distance', 'distance_rate']


#statis feature+ leven_distance feature
X2=np.load('./data_word2vec_feature/train_dnn_data3_feature.npz')
X_val_2=np.load('./data_word2vec_feature/val_dnn_data3_feature.npz')
X_test_2=np.load('./data_word2vec_feature/test_dnn_data3_feature.npz')

for i in [0,1,2,26,27,28]:
    X=np.concatenate((X,np.reshape(X2['train_data'][:,i],(-1,1))),axis=1)
    X_val_=np.concatenate((X_val_,np.reshape(X_val_2['val_data'][:,i],(-1,1))),axis=1)
    X_test_=np.concatenate((X_test_,np.reshape(X_test_2['test_data'][:,i],(-1,1))),axis=1)


#feature_names=['prefix_cut_in_title', 'prefix_cut_in_start_title',
#       'query_prediction_length', 'prefix_count', 'prefix_click',
#       'prefix_ctr', 'title_click', 'title_count', 'title_ctr',
#       'tag_click', 'tag_count', 'tag_ctr', 'prefix_title_click',
#       'prefix_titlecount', 'prefix_title_ctr', 'prefix_tag_click',
#       'prefix_tagcount', 'prefix_tag_ctr', 'title_tag_click',
#       'title_tagcount', 'title_tag_ctr', 'prefix_title_tag_click',
#       'prefix_title_tagcount', 'prefix_title_tag_ctr',
#       'prefix_tag_nunique', 'title_tag_nunique', 'is_in_title',
#       'leven_distance', 'distance_rate']
#
#y=np.reshape(np.array(X['y_train']),(-1,))
#y_val=np.reshape(np.array(X_val_['y_val']),(-1,))
##y=np.reshape(np.concatenate((y,y_val),axis=0),(-1,))
#X=np.array(X['train_data'])
#X_val_=np.array(X_val_['val_data'])
##X=np.concatenate((X,X_val_),axis=0)
#X_test_=np.array(X_test_['test_data'])

print('train beginning')

#X = np.array(train_data_.drop(['label'], axis = 1))
#y = np.array(train_data_['label'])
#X_test_ = np.array(test_data_.drop(['label'], axis = 1))
print('================================')
print(X.shape)
print(y.shape)
print('================================')


xx_logloss = []
xx_submit = []
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

#params = {
#    'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'metric': 'binary_logloss',
#    'num_leaves': 32,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 1
#}
params = {
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 6000,
    "max_bin": 425,
    "subsample_for_bin": 20000,
    "objective": 'binary',
    "min_split_gain": 0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 1,
    "reg_alpha": 3,
    "reg_lambda": 5,
    "seed": 2018,
    "n_jobs": 5,
    "verbose": 1,
    "silent": False
}

best_f1_score=0
#for k, (train_in, test_in) in enumerate(skf.split(X, y)):
#    print('train _K_ flod', k)
#    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
X_train, X_test, y_train, y_test=X,X_val_,y,y_val
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=100,
                verbose_eval=50,
                )
#feature importance
feature_imp=pd.DataFrame({'column': feature_names,'importance': gbm.feature_importance()})

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

test_data_['label'] = list(s / 1)
test_data_['label'] = test_data_['label'].apply(lambda x: round(x))
print('test_logloss:', np.mean(test_data_.label))
test_data_['label']=test_data_['label'].apply(lambda x:int(x))
test_data_['label'].to_csv('./submit/result_baseline.csv',index = False)

val_df=pd.DataFrame(np.where(val_test>0.4, 1,0),columns=['label'])
val_df['label']=val_df['label'].apply(lambda x:int(x))
val_df['label'].to_csv('./submit/result_val_baseline.csv',index = False)