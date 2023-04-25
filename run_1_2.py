# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:03:31 2023

@author: biraaj
"""

from functions import load_data_softmax_train,softmax_regression,bagging,adaboost
import numpy as np

program_data_main = "./data/program_data_2c_2d_2e_hw1.txt"
alpha = 0.01
epoch = 3200

features,targets,unique_classes = load_data_softmax_train(program_data_main,0)

split_ratio = int(features.shape[0] * 0.8)

X_train = features[:split_ratio]
Y_train = targets[:split_ratio]

X_test = features[split_ratio:]
Y_test = targets[split_ratio:]


## 1 bagging

print("Results of 1a,b ################################ ")
softmax_model = softmax_regression(epoch,alpha)
softmax_model.softmax_regression(X_train,Y_train,unique_classes)
Y_pred = softmax_model.softmax_predict(X_test)
print("Accuracy on single classifier = ",np.mean(np.argmax(Y_test,axis=1) == Y_pred))


for i in [10,50,100]:
    print("bagging with ",i,"estimators = ",bagging(X_train,Y_train,X_test,Y_test,i,unique_classes))

print("################################")
print()


## 2 boosting

print("Results of 2a,b ################################ ")
alpha = 0.1
epoch = 3600

softmax_model = softmax_regression(3200,0.01)
softmax_model.softmax_regression(X_train,Y_train,unique_classes)
Y_pred = softmax_model.softmax_predict(X_test)
print("Accuracy on single classifier = ",np.mean(np.argmax(Y_test,axis=1) == Y_pred))


for i in [10,50,100]:
     print("boosting with ",i,"estimators, accuarcy = ",adaboost(X_train,Y_train,X_test,Y_test,i,unique_classes,alpha,epoch))
     
print("################################")
print()