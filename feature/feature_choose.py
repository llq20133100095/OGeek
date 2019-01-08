#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:21:09 2018

@author: llq
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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

def tag_group(train_data,test_data):
    """
    针对tag标签进行分割
    """
    data=pd.concat([train_data,test_data])
    items = ['prefix', 'title']
    target=['tag']
    for i in items:
        temp=data.groupby(i,as_index = False)[target[0]].agg({i+"_"+target[0]+"_nunique":'count'})
        train_data = pd.merge(train_data, temp, on=i, how='left')
        test_data = pd.merge(test_data, temp, on=i, how='left')
    
#    temp=data.groupby(target[0],as_index = False)[items[0]].agg({target[0]+"_"+items[0]+"_nunique":'count'})
#    train_data = pd.merge(train_data, temp, on=target[0], how='left')
#    test_data = pd.merge(test_data, temp, on=target[0], how='left')
#
#    temp=data.groupby(target[0],as_index = False)[items[1]].agg({target[0]+"_"+items[1]+"_nunique":'count'})
#    train_data = pd.merge(train_data, temp, on=target[0], how='left')
#    test_data = pd.merge(test_data, temp, on=target[0], how='left')
    
    return train_data,test_data
    
def get_data():
    
    def read_file(path,is_test=False):
        fp = open(path)
        dataset = []
        for line in fp.readlines():
            line = line.strip().split('\t')
            if is_test:
                line.append(-1)
            dataset.append(line)
        data = pd.DataFrame(dataset)
        data.columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']
        return data

    """
    baseline feature
    """
    train_data = read_file('../data/oppo_round1_train_20180929.txt').astype(str)
    
    val_data = read_file('../data/oppo_round1_vali_20180929.txt').astype(str)
    test_data = read_file('../data/oppo_round1_test_A_20180929.txt',True).astype(str)
    
    train_data = train_data[train_data['label'] != '音乐' ]
    test_data['label'] = -1
    
    train_data = pd.concat([train_data,val_data])
    train_data['label'] = train_data['label'].apply(lambda x: int(x))
    test_data['label'] = test_data['label'].apply(lambda x: int(x))
    items = ['prefix', 'title', 'tag']
    
    #构造长度特征
    train_data=length_feature(train_data)
    test_data=length_feature(test_data)

    #单个特征
    for item in items:
        #通过item对label进行分割
        temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
        #统计item的点击次数
        temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count']+0.001)
        train_data = pd.merge(train_data, temp, on=item, how='left')
        test_data = pd.merge(test_data, temp, on=item, how='left')
    #交叉特征
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item_g = [items[i], items[j]]
            temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
            temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+0.001)
            train_data = pd.merge(train_data, temp, on=item_g, how='left')
            test_data = pd.merge(test_data, temp, on=item_g, how='left')
    #3 feature across
    item_g = [items[0], items[1], items[2]]
    temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
    temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
    train_data = pd.merge(train_data, temp, on=item_g, how='left')
    test_data = pd.merge(test_data, temp, on=item_g, how='left')

    #针对tag标签进行分割
    train_data,test_data=tag_group(train_data,test_data)
    
    #delete the column 
    train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
    test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
#    train_data_=train_data_[['prefix_ctr','title_ctr','tag_ctr','prefix_title_ctr','prefix_tag_ctr','title_tag_ctr','label']]
#    test_data_=test_data_[['prefix_ctr','title_ctr','tag_ctr','prefix_title_ctr','prefix_tag_ctr','title_tag_ctr','label']]
#    train_data_=train_data_[['tag_ctr','label']]
#    test_data_=test_data_[['tag_ctr','label']]
    
#    X = np.array(train_data_.drop(['label'], axis = 1))
#    y = np.array(train_data_['label'])
#    X_test_ = np.array(test_data_.drop(['label'], axis = 1))

    return train_data_,test_data_


train_data_,test_data_=get_data()

#print pearsonr(train_data_[['query_prediction_length']],train_data_[['label']])
#print pearsonr(train_data_[['prefix_count']],train_data_[['label']])
#print pearsonr(train_data_[['prefix_click']],train_data_[['label']])
#print pearsonr(train_data_[['prefix_ctr']],train_data_[['label']])

##选择K个最好的特征，返回选择特征后的数据
#skb = SelectKBest(chi2, k=5)
#X_new=skb.fit_transform(X, y)
#X_test_new=X_test_[:,skb.get_support(True)]