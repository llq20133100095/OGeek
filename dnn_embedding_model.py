#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:33:00 2018

@author: llq
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from dnn_data import get_data,process_data,onehot_feature,iterate_minibatches3
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
tf.reset_default_graph()

class DNN():
    def __init__(self,voc_size=1,tag_size=24,K=None):
        self.voc_size=voc_size
        self.tag_size=tag_size
        self.embed_size=22
        self.logdir='./summary'
        self.epochs=20
        self.batch_size=600
        self.N=5
        self.model_name="dnn_"+str(self.batch_size)
        self.submit_file="./submit/dnn/result_dnn.csv"
        self.regularizer=0.001
        self.K=K
        
        with tf.name_scope("inputs"):
            self.target=tf.placeholder(tf.float32, shape=(None, 1), name="target")
            
            self.embedding=tf.placeholder(tf.float32, shape=(None, self.K), name="embedding_input")
            self.feature=tf.placeholder(tf.float32, shape=(None, 24), name="feature_input")
            self.dropout=tf.placeholder(tf.float32, name="dropout")
                
    def dnn_embedding_model(self,X_emb,X_fea,y,X_test_emb,X_test_fea,y_test):
        """
        1.dnn
        2.dropout+l2
        3.embedding+feature
        """
        with tf.name_scope("dense"):
            dense_embedding = tf.layers.dense(inputs=self.embedding, units=100, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            dense=tf.layers.dropout(dense, rate=self.dropout, training=True)
            dense_feature = tf.layers.dense(inputs=self.feature, units=100, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            
            dense_con=tf.concat((dense_embedding,dense_feature),axis=1)
            
            dense = tf.layers.dense(inputs=dense_con, units=300, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense = tf.layers.dense(inputs=dense, units=200, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
#        with tf.name_scope("attention"):
#            w_att = tf.get_variable('wight_att', [1, 100], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            b_att = tf.get_variable('b_att', [100], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#            u_att = tf.get_variable('u_att', [100,1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), \
#                regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
#
#            H=tf.reshape(dense,(-1,1))
#            M=tf.tanh(tf.matmul(H, w_att)+b_att)
#            alpha=tf.nn.softmax(tf.reshape(tf.matmul(M,u_att),(-1,dense.shape[1])),dim=1)
#            y_pred=tf.reshape(tf.reduce_sum(dense * alpha, axis=1),(-1,1))
            
            y_pred=tf.layers.dense(inputs=dense,units=1,activation=tf.nn.sigmoid,\
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
        
        with tf.name_scope("Loss"):
            logloss=-tf.reduce_mean(tf.log(y_pred)*self.target+(1-self.target)*tf.log(1-y_pred)) \
                +tf.losses.get_regularization_loss()
#            label_onehot=tf.one_hot(indices=self.target, depth=2, dtype=tf.float32)
#            logloss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=y_pred)) \
#                +tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', logloss)
        
        with tf.name_scope("training_op"):
            global_step = tf.placeholder(dtype=tf.int32)
            learning_rate = tf.train.exponential_decay(1e-4,global_step,decay_steps=2,decay_rate=1.0,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(logloss)
        
        # Summary
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        
        #init
        init=tf.global_variables_initializer()
    
        #split train_data: train and val:42
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for (train_in, test_in) in skf.split(X, y):
            X_train_emb, X_train_fea, X_val_emb, X_val_fea, y_train, y_val = X_emb[train_in], X_fea[train_in], X_emb[test_in], X_fea[test_in], y[train_in], y[test_in]
            break
        del X_emb,X_fea,y
        
        best_f1_score=0
        best_epoch=0
        ypred_test=0
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                start=time.time()
                #train
                loss_epoch=[]
                for batch in iterate_minibatches3(X_train_emb, X_train_fea, y_train, self.batch_size, shuffle=True):
                    train_emb, train_fea, traget = batch
                    feed_dict_train = {
                        self.embedding: train_emb,
                        self.feature: train_fea,
                        self.dropout:0.5,
                        self.target: traget,
                        global_step:epoch,
                    }
                    
                    sess.run(training_op,feed_dict_train)
                    e,summary,lr,out=sess.run([logloss,merged_summary,learning_rate,y_pred],feed_dict_train)
                    loss_epoch.append(e)
                    
                summary_writer.add_summary(summary,epoch)
                print("epoch: %d | lr: %f" % (epoch,lr))
                print("train loss: %f" % np.mean(loss_epoch))
                
                
                
                #val:
                loss_epoch=[]
#                y_pre_val=np.array([[0]])
#                for batch in iterate_minibatches2(X_val, y_val, self.batch_size, shuffle=False):
#                    train, traget = batch
                feed_dict_val = {
                    self.embedding: X_val_emb,
                    self.feature: X_val_fea,
                    self.dropout:0.0,
                    self.target: y_val,
                }
                    
                e=logloss.eval(feed_dict=feed_dict_val)
                loss_epoch.append(e)
                ypred_val=sess.run(y_pred,feed_dict=feed_dict_val)
#                y_pre_val=np.concatenate((y_pre_val,np.reshape(ypred_val,(-1,1))),axis=0)
                
#                y_pre_val=y_pre_val[1:]
                f1_sco=f1_score(y_val,np.where(ypred_val>0.4,1,0))
                print("val loss: %f" % np.mean(loss_epoch))
                print("f1_score: %f" % f1_sco)
                
                #test
                if(best_f1_score<f1_sco):
                    best_f1_score=f1_sco
                    best_epoch=epoch
                    feed_dict_test = {
                        self.embedding: X_test_emb,
                        self.feature: X_test_fea,
                        self.dropout:0.0,
                        self.target: y_test,
                    }
                    ypred_test1=sess.run(y_pred,feed_dict=feed_dict_test)
                    ypred_test=np.where(ypred_test1>0.4,1,0)
                    ypred_test=pd.DataFrame(ypred_test,columns=['label'])
                    ypred_test['label']=ypred_test['label'].apply(lambda x:int(x))
                    ypred_test['label'].to_csv(self.submit_file,index = False)
 
                    
                saver = tf.train.Saver()
                saver.save(sess, "./checkpoint_dir/MyModel_"+self.model_name, global_step=epoch)
                print("time: %f s" % (time.time()-start))
                
            print("best_f1_score: %f" % best_f1_score)
            print("best_epoch: %d" % best_epoch)
               
            return ypred_test1

"""
5.embedding+feature
"""
X_train=np.load('./data/train_vector.npz')
X_test=np.load('./data/test_vector.npz')

X=X_train['X_train']
y=X_train['y_train']

X_test_=X_test['X_test']
y_test=X_test['y_test']

X2,_,X_test_2,_=get_data()
#chang nan to 0
X2=np.nan_to_num(X2) 
X_test_2=np.nan_to_num(X_test_2)

for i in range(len(X2[0])):
    if(X2[0,i]>1):
        se=X2[:,i]
        semax=se.max()
        semin=se.min()
        X2[:,i]=(se-semin)/(semax-semin)
        se_text=X_test_2[:,i]
        X_test_2[:,i]=(se_text-se_text.min())/(se_text.max()-se_text.min())

print("begin training")
dnn=DNN(K=622)
out=dnn.dnn_embedding_model(X, X2, y, X_test_, X_test_2, y_test)
            
