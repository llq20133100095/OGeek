#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:15:49 2018

@author: llq
"""
import pandas as pd
from sklearn.metrics import f1_score
from dnn_data3 import char_lower

def split():
    """
    split the val_data and test_data in hasdata and nodata.
    """
    train_data = read_file('../data/oppo_round1_train_20180929.txt').astype(str)
    val_data = read_file('../data/oppo_round1_vali_20180929.txt').astype(str)
    test_data = read_file('../data/oppo_round1_test_A_20180929.txt',True).astype(str)
    
    train_data = train_data[train_data['label'] != '音乐' ]
    test_data['label'] = -1
    
    train_data['label'] = train_data['label'].apply(lambda x: int(x))
    val_data['label'] = val_data['label'].apply(lambda x: int(x))
    test_data['label'] = test_data['label'].apply(lambda x: int(x))
    
    #lower
    train_data["prefix"] = train_data["prefix"].apply(char_lower)
    train_data["title"] = train_data["title"].apply(char_lower)
    val_data['prefix'] = val_data['prefix'].apply(char_lower)
    val_data['title'] = val_data['title'].apply(char_lower)
    test_data["prefix"] = test_data["prefix"].apply(char_lower)
    test_data['title'] = test_data['title'].apply(char_lower)
    
    #split: has data in train_data or no data in train_data
    items=['prefix', 'title', 'tag']
    train_data=train_data.drop_duplicates(items)
    val_merge=pd.merge(val_data,train_data,on=items,how='left')
    test_merge=pd.merge(test_data,train_data,on=items,how='left')
    
    val_hasdata=val_merge[pd.isnull(val_merge['label_y'])==False]
    test_hasdata=test_merge[pd.isnull(test_merge['label_y'])==False]
    
    val_notdata=val_merge[pd.isnull(val_merge['label_y'])]
    test_notdata=test_merge[pd.isnull(test_merge['label_y'])]
    
    val_hasdata_index=list(val_hasdata.index)
    test_hasdata_index=list(test_hasdata.index)
    
    val_notdata_index=list(val_notdata.index)
    test_notdata_index=list(test_notdata.index)
    
    return val_hasdata_index,test_hasdata_index,val_notdata_index,test_notdata_index,val_hasdata,val_notdata

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

train_data = read_file('../data/oppo_round1_train_20180929.txt').astype(str)
val_data = read_file('../data/oppo_round1_vali_20180929.txt').astype(str)
train_data['label'] = train_data['label'].apply(lambda x: int(x))
val_data['label'] = val_data['label'].apply(lambda x: int(x))

#lower
train_data["prefix"] = train_data["prefix"].apply(char_lower)
train_data["title"] = train_data["title"].apply(char_lower)
val_data['prefix'] = val_data['prefix'].apply(char_lower)
val_data['title'] = val_data['title'].apply(char_lower)

items=['prefix','title','tag']

#3 feature across
item_g = [items[0], items[1], items[2]]
temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
val_data = pd.merge(val_data, temp, on=item_g, how='left')
    

val_save=pd.read_csv('../submit/split_data/val_save.csv')

#hasdata and notdata in val_data and test_data
val_hasdata_index,_,val_notdata_index,_,_,_=split()
val_hasdata=val_data.ix[val_hasdata_index]

#1.hasdata
val_hasdata.reset_index(inplace=True)
val_hasdata['label_y']=val_save['label']

a_false=val_hasdata[val_hasdata['label_y']!=val_hasdata['label']]
a_true=val_hasdata[val_hasdata['label_y']==val_hasdata['label']]
b=a_false[a_false['prefix_title_tag_ctr']>0.3]

val_hasdata_large3=val_hasdata[val_hasdata['prefix_title_tagcount']>1]
val_hasdata_less3=val_hasdata[val_hasdata['prefix_title_tagcount']<=1]

large3_true=len(val_hasdata_large3[val_hasdata_large3['label_y']==val_hasdata_large3['label']])
large3_rate=float(large3_true)/len(val_hasdata_large3)

less3_true=len(val_hasdata_less3[val_hasdata_less3['label_y']==val_hasdata_less3['label']])
less3_rate=float(less3_true)/len(val_hasdata_less3)

#2.nodata
val_nodata_save=pd.read_csv('../submit/split_data/val_nodata_save.csv')

val_nodata=val_data.ix[val_notdata_index]
val_nodata.reset_index(inplace=True)
val_nodata['label_y']=val_nodata_save['label']

val_nodata_false=val_nodata[val_nodata['label']!=val_nodata['label_y']]
