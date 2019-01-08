#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:54:56 2018

@author: llq
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
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

train_data = read_file('./data/oppo_round1_train_20180929.txt').astype(str)
val_data = read_file('./data/oppo_round1_vali_20180929.txt').astype(str)
test_data = read_file('./data/oppo_round1_test_A_20180929.txt',True).astype(str)

train_data = train_data[train_data['label'] != '音乐' ]
test_data['label'] = -1

#train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))


#get the relu:items
items = ['prefix', 'title', 'tag']
relu=train_data[:-50000].groupby(items,as_index=False)['label'].agg({'3_feature_click': 'sum','3_feature_count':'count'})
relu['3_feature_ctr'] = relu['3_feature_click']/(relu['3_feature_count']+0.01)

#"""
#parameter choose in val data
#"""
##when the ctr>0.7,we think this label is 1.
#best_f1=0
#for i in np.arange(0.3,0.5,0.02):
#    threshold=0.0
#    train_data1 = pd.merge(train_data[:-50000], relu, on=items, how='left')
#    train_relu=train_data1[train_data1['3_feature_ctr']>threshold]
#    train_relu=train_relu[train_relu['label']==1]
#    train_relu=train_relu[['prefix', 'title', 'tag','label','3_feature_ctr']]
#    train_relu=train_relu.drop_duplicates()
#    
#    #Use the rule in test_data
#    val_relu = pd.merge(val_data, train_relu, on=items, how='left')
#    val_relu=val_relu.fillna("0")
#    val_relu['label_y']=val_relu['label_y'].apply(lambda x:int(x))
#    
#    #combine GBDT and Statis data "1"
#    val_relu=pd.DataFrame(val_relu['label_y'])
#    val_dnn=pd.read_table('./submit/split_data/result_val.csv',names= ['label_y'], header= None).astype(int)
#    val_dnn=val_relu+val_dnn
#    val_dnn=pd.DataFrame(val_dnn['label_y'].apply(lambda x: np.where(x>1,1,x)))
#    
#    '''
#    2.label 0
#    '''
#    zero_threshold=i
#    train_data2 = pd.merge(train_data[:-50000], relu, on=items, how='left')
#    zero_relu=train_data2[train_data2['3_feature_ctr']<zero_threshold]
#    zero_relu=zero_relu[zero_relu['label']==0]
#    zero_relu=zero_relu[['prefix', 'title', 'tag','label','3_feature_ctr']]
#    zero_relu=zero_relu.drop_duplicates()
#    
#    #Use the rule in test_data
#    val_relu = pd.merge(val_data, zero_relu, on=items, how='left')
#    val_relu=val_relu.fillna("1")
#    val_relu['label_y']=val_relu['label_y'].apply(lambda x:int(x))
#    
#    val_relu=pd.DataFrame(val_relu['label_y'])
#    val_combine=val_relu + val_dnn
#    val_combine=pd.DataFrame(val_combine['label_y'].apply(lambda x: np.where(x==2,1,0)))
#    
#    f1=f1_score(np.array(val_data['label'],dtype=np.float),val_combine['label_y'])
#    print f1
#    if(best_f1<f1):
#        best_f1=f1
#        print i


"""
1.label 1
"""
#when the ctr>0.7,we think this label is 1.
threshold=0.0
train_data1 = pd.merge(train_data[:-50000], relu, on=items, how='left')
train_relu=train_data1[train_data1['3_feature_ctr']>threshold]
train_relu=train_relu[train_relu['label']==1]
train_relu=train_relu[['prefix', 'title', 'tag','label','3_feature_ctr']]
train_relu=train_relu.drop_duplicates()
    
#Use the rule in test_data
test_relu = pd.merge(test_data, train_relu, on=items, how='left')
test_relu=test_relu.fillna("0")
test_relu['label_y']=test_relu['label_y'].apply(lambda x:int(x))

#combine GBDT and Statis data "1"
test_relu=pd.DataFrame(test_relu['label_y'])
test_dnn=pd.read_table('./submit/split_data/result1_0.735895.csv',names= ['label_y'], header= None).astype(int)
test_dnn=test_relu+test_dnn
test_dnn=pd.DataFrame(test_dnn['label_y'].apply(lambda x: np.where(x>1,1,x)))
#test_combine['label_y'].to_csv('./submit/statis/result_GBDT_statis_'+str(threshold)+'.csv',index = False)

'''
2.label 0
'''
zero_threshold=0.36
train_data2 = pd.merge(train_data[:-50000], relu, on=items, how='left')
zero_relu=train_data2[train_data2['3_feature_ctr']<zero_threshold]
zero_relu=zero_relu[zero_relu['label']==0]
zero_relu=zero_relu[['prefix', 'title', 'tag','label','3_feature_ctr']]
zero_relu=zero_relu.drop_duplicates()

#Use the rule in test_data
test_relu = pd.merge(test_data, zero_relu, on=items, how='left')
test_relu=test_relu.fillna("1")
test_relu['label_y']=test_relu['label_y'].apply(lambda x:int(x))

test_relu=pd.DataFrame(test_relu['label_y'])
#test_dnn=pd.read_table('./submit/dnn/result_dnn_0.802552.csv',names= ['label_y'], header= None).astype(int)
test_combine=test_relu + test_dnn
test_combine=pd.DataFrame(test_combine['label_y'].apply(lambda x: np.where(x==2,1,0)))
test_combine['label_y'].to_csv('./submit/statis/result_baseline_0.735895_zero_one'+str(zero_threshold)+'.csv',index = False)
