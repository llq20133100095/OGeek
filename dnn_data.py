#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:06:44 2018

@author: llq
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import sparse

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
#    train_data = pd.read_table('./data/oppo_round1_train_20180929.txt', 
#            names= ['prefix','query_prediction','title','tag','label'], header= None).astype(str)
#    
#    val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt', 
#            names = ['prefix','query_prediction','title','tag','label'], header = None).astype(str)
#    test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',
#            names = ['prefix','query_prediction','title','tag'],header = None).astype(str)

    train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)
    
    val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
    test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)
    
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
    
    X = np.array(train_data_.drop(['label'], axis = 1))
    y = np.array(train_data_['label'])
    X_test_ = np.array(test_data_.drop(['label'], axis = 1))

    return X,y,X_test_,test_data_

def read_data():
    train_data = pd.read_table('./data/oppo_round1_train_20180929.txt', 
            names= ['prefix','query_prediction','title','tag','label'], header= None).astype(str)
    
    val_data = pd.read_table('./data/oppo_round1_vali_20180929.txt', 
            names = ['prefix','query_prediction','title','tag','label'], header = None).astype(str)
    test_data = pd.read_table('./data/oppo_round1_test_A_20180929.txt',
            names = ['prefix','query_prediction','title','tag'],header = None).astype(str)
    
    train_data = train_data[train_data['label'] != '音乐' ]
    test_data['label'] = -1

    return train_data,val_data,test_data

def process_data():
    """
    id feature
    """
    train_data,val_data,test_data=read_data()
    
    train_data = pd.concat([train_data,val_data])
    train_data['label'] = train_data['label'].apply(lambda x: int(x))
    test_data['label'] = test_data['label'].apply(lambda x: int(x))
    data=pd.concat([train_data,test_data])
    items = ['prefix', 'title', 'tag']
    
    
    #change the 'prefix' and 'title' to id.
    prefix_statis=data.groupby(['prefix'], as_index = False)['label'].agg({'prefix_count':'count'})
    prefix_statis['voc_dic']=prefix_statis['prefix']
    
    title_statis=data.groupby(['title'], as_index = False)['label'].agg({'title_count':'count'})
    title_statis['voc_dic']=title_statis['title']
    
    voc_dic=pd.merge(prefix_statis,title_statis,on=['voc_dic'],how='outer')
    voc_dic['voc_id']=voc_dic['voc_dic'].index
    voc_dic=voc_dic.drop(['prefix','prefix_count','title','title_count'],axis=1)
    
    merge_dic=voc_dic
    for i in items[:2]:
        merge_dic=voc_dic.copy()
        merge_dic[i]=merge_dic['voc_dic']
        merge_dic[i+"_id"]=merge_dic['voc_id']
        merge_dic=merge_dic.drop(['voc_dic','voc_id'],axis=1)
        train_data = pd.merge(train_data, merge_dic, on=i, how='left')
        test_data = pd.merge(test_data, merge_dic, on=i, how='left')
        
    #change the 'title' to id
    tag_statis=data.groupby(['tag'], as_index = False)['label'].agg({'tag_count':'count'})
    tag_statis['tag_id']=tag_statis['tag'].index
    tag_statis=tag_statis.drop(['tag_count'],axis=1)
    train_data = pd.merge(train_data, tag_statis, on=['tag'], how='left')
    test_data = pd.merge(test_data, tag_statis, on=['tag'], how='left')
    
    #embedding size
    voc_size=len(voc_dic)
    #tag size
    tag_size=len(tag_statis)
    
    return train_data,test_data,voc_size,tag_size

def onehot_feature(train_data_,test_data_):
    """
    Change the tag_id to one-hot
    """
    one_hot_feature=['tag_id'] 
    data=pd.concat([train_data_,test_data_])
    enc = OneHotEncoder()
    enc.fit(data[one_hot_feature[0]].values.reshape(-1,1))
    train_tag=enc.transform(train_data_[one_hot_feature[0]].values.reshape(-1, 1))
    test_tag=enc.transform(test_data_[one_hot_feature[0]].values.reshape(-1, 1))
    print('one-hot prepared !')
    return train_tag,test_tag


def iterate_minibatches(X_prefix, X_title, X_tag, y, batchsize, shuffle=False):
        """
        Get minibatches
        """
        assert len(X_prefix) == len(y)
        if shuffle:
            indices = np.arange(len(X_prefix))
            np.random.shuffle(indices)
        for start_idx in range(0, len(X_prefix) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield X_prefix[excerpt], X_title[excerpt], X_tag[excerpt], y[excerpt]


def iterate_minibatches2(X, y, batchsize, shuffle=False):
        """
        Get minibatches
        """
        assert len(X) == len(y)
        if shuffle:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
        for start_idx in range(0, len(X) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]
            
def iterate_minibatches3(X_emb, X_fea, y, batchsize, shuffle=False):
        """
        Get minibatches
        """
        assert len(X_emb) == len(y)
        if shuffle:
            indices = np.arange(len(X_emb))
            np.random.shuffle(indices)
        for start_idx in range(0, len(X_emb) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield X_emb[excerpt], X_fea[excerpt], y[excerpt]


if __name__ == "__main__":
#    """
#    1.
#    prefix and title -> id
#    tag -> one-hot
#    """
#    train_data,test_data,voc_size,tag_size=process_data()
#    
#    #delete the column 
#    train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
#    test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
#    
#    #title_id to one-hot
#    X_tag,X_test_tag=onehot_feature(train_data_,test_data_)
#    
#    X_prefix=np.array(train_data_['prefix_id'])
#    X_title=np.array(train_data_['title_id'])
#    X_tag=X_tag.toarray()
#    y=np.array(train_data_['label'])
#    
#    X_test_prefix=np.array(test_data_['prefix_id'])
#    X_test_title=np.array(test_data_['title_id'])
#    X_test_tag=X_test_tag.toarray()
    
    """
    3.all feature -> normalization(0,1)
    """
    feature=[3,6,9,12,15,18,21]
    X,y,X_test_,test_data_=get_data()
    
    #chang nan to 0
    X=np.nan_to_num(X) 
    X_test_=np.nan_to_num(X_test_)
    
    #normalization
    for i in range(len(X[0])):
        if(i not in feature):
            se=X[:,i]
            semax=se.max()
            semin=se.min()
            X[:,i]=(se-semin)/(semax-semin)
            se_text=X_test_[:,i]
            X_test_[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())
    
    y=np.reshape(y,(-1,1))
    y_test=np.array(test_data_[['label']])