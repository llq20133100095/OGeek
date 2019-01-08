#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:53:17 2018

@author: llq
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def smote(X, y, max_amount=0, std_rate=5, kneighbor=5, kdistinctvalue=10, method='mean'):
    start=time.time()
    y=np.reshape(y,(-1,1))
    data=np.concatenate((X,y),axis=1)
    try:
        data = pd.DataFrame(data)
    except:
        raise ValueError
    tag_index=-1
    case_state = data.iloc[:, tag_index].groupby(data.iloc[:, tag_index]).count()
    case_rate = max(case_state) / min(case_state)
    location = []
    if case_rate < 0.5:
        print case_rate
        print('不需要smote过程')
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == min(case_state)].index)[0]])
#        more_data = np.array(
#            data[data.iloc[:, tag_index] == np.array(case_state[case_state == max(case_state)].index)[0]])
        # 找出每个少量数据中每条数据k个邻居
        neighbors = NearestNeighbors(n_neighbors=kneighbor).fit(less_data)
        less_data_choose=less_data
        for i in range(len(less_data_choose)):
            location_set = neighbors.kneighbors([less_data[i]], return_distance=False)[0]
            location.append(location_set)
            if(i%50000==0):
                print('it processes %s in %s less_data' % (i+1,less_data_choose.shape[0]))
            
        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数，如果没有按照std_rate(预期正负样本比)比例生成
        if max_amount > 0:
            amount = max_amount
        else:
            amount = int(max(case_state) / std_rate)
        
        
        times = 0
        while times<amount:
            #shuffle:choose a neighbor data
            ran=np.random.randint(1,kneighbor)
            ran_loc=np.random.randint(0,len(location))
#            np.random.shuffle(location)
            less_data_index=location[ran_loc][ran]
            center_index=location[ran_loc][0]
            
            #create new data. And label set to 1.
            gap=np.random.random()
            dif=less_data[center_index]-less_data[less_data_index]
            new_case=less_data[center_index]+gap*abs(dif)
            
            #set label to 1
            new_case=np.reshape(new_case,(1,-1))
            new_case[0,-1]=1
            if times==0:
                update_case=new_case
            elif times!=0:
                update_case=np.concatenate((update_case,new_case),axis=0)
            
#            if(times%50000==0):
            print('it cretes %s new data，completeing %.2f' % (times+1, times * 100 / amount))
            times = times + 1              
            
            # 保存处理后的数据集
            if(times%10000==0):
                #concate
                data_save=np.array(data)
                data_save=np.concatenate((data_save,update_case),axis=0)
                
                #train_y
                y_train=data_save[:,-1]
                #train_x
                X=data_save[:,:-1]
                
                data_save = dict(
                    train_data=X,
                    y_train=y_train,
                )
                np.savez('./data/smote/train_data_epoch_times.npz', **data_save)
    
        #concate
        data=np.array(data)
        data=np.concatenate((data,update_case),axis=0)
        
        #train_y
        y_train=data[:,-1]
        #train_x
        X=data[:,:-1]
    
    print("time: %f s" %(time.time()-start))
    return X,y_train

def repeate_smote(X, y, max_amount=0, std_rate=5, kneighbor=5, kdistinctvalue=10, method='mean'):
    start=time.time()
    y=np.reshape(y,(-1,1))
    data=np.concatenate((X,y),axis=1)
    try:
        data = pd.DataFrame(data)
    except:
        raise ValueError
    tag_index=-1
    case_state = data.iloc[:, tag_index].groupby(data.iloc[:, tag_index]).count()
    case_rate = max(case_state) / min(case_state)

    if case_rate < 0.5:
        print case_rate
        print('不需要smote过程')
        return data
    else:
        # 拆分不同大小的数据集合
        less_data = np.array(
            data[data.iloc[:, tag_index] == np.array(case_state[case_state == min(case_state)].index)[0]])

        # 确定需要将少量数据补充到上限额度
        # 判断有没有设定生成数据个数，如果没有按照std_rate(预期正负样本比)比例生成
        if max_amount > 0:
            amount = max_amount
        else:
            amount = int(max(case_state) / std_rate)
    
        times = 0
        while times<amount:
            #shuffle:choose a neighbor data
            ran=np.random.randint(0,less_data.shape[0])

            
            #create new data. And label set to 1.
            new_case=less_data[ran]
            
            #set label to 1
            new_case=np.reshape(new_case,(1,-1))
            new_case[0,-1]=1
            if times==0:
                update_case=new_case
            elif times!=0:
                update_case=np.concatenate((update_case,new_case),axis=0)
            
#            if(times%50000==0):
            print('it cretes %s new data，completeing %.2f' % (times+1, times * 100 / amount))
            times = times + 1              

        #concate
        data=np.array(data)
        data=np.concatenate((data,update_case),axis=0)
        
        #train_y
        y_train=data[:,-1]
        #train_x
        X=data[:,:-1]
    
    print("time: %f s" %(time.time()-start))
    return X,y_train
    
if __name__ == "__main__":
    #statis feature
    X=np.load('./data_word2vec_feature/train_dnn_data3_feature.npz')
#    X_val_=np.load('./data_word2vec_feature/val_dnn_data3_feature.npz')
#    X_test_=np.load('./data_word2vec_feature/test_dnn_data3_feature.npz')
    
    y=np.array(X['y_train'])
    X=np.array(X['train_data'])
    
    X2,y2=smote(X,y,max_amount=200000)
    
#    # 保存处理后的数据集
#    data = dict(
#        train_data=X2,
#        y_train=y2,
#    )
#    np.savez('./data/smote/train_data.npz', **data)
    
        # 保存处理后的数据集
    data = dict(
        train_data=X2,
        y_train=y2,
    )
    np.savez('./data/smote/train_data_20w.npz', **data)
