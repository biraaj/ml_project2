# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:04:05 2023

@author: biraaj
"""

import numpy as np

## Used the below functions from my project 1 functions.py.

def replace_parenthesis(_string):
    """
        Function to filter out data and split by removing parenthesis.
        Note: This implmentation was from hw1
    """
    return _string.replace('(','').replace(')','').replace(' ','').strip().split(",")


##### Softmax Regression ##########################################################

## Used the below functions from my project 1 run_3a_b_c.py.
def load_data_softmax_train(train_data_path,remove_from_end=0):
    """
        This is a function to load the training data for classification problems required for softmax regression.
        This returns the feature array with bias and target array by hot encoding it.
    """
    _feat = []
    _target = []
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-1-remove_from_end)]])
            _target.append(temp_data[len(temp_data)-1])

    unique_labels = list(set(_target))
    target_array = np.array(_target)
    single_hot_array = np.zeros((len(_target),len(unique_labels)))
    for _index,_val in enumerate(unique_labels):
        single_hot_array[:,_index] =  (target_array == _val).astype(int)
        
    #adding bias term to feature array
    _feat = np.insert(np.array(_feat),0,1,axis=1) 
    
    
    return _feat,single_hot_array,unique_labels



class softmax_regression:
    ## Used the below functions from my project 1  functions.py and run_3a_b_c.py and modified them.
    def __init__(self,epochs=3200,learning_rate=0.01):
        self.weights = []
        self.epochs = epochs
        self.learning_rate =  learning_rate
    
    def softmax_regression(self,train_features,train_targets,unique_classes,boosting=False,sample_weight=None):
        #Used the function from project 1 run_3a_b_c.py
        self.weights = np.zeros((train_features.shape[1],len(unique_classes)))
        for _epoch in range(self.epochs):
            feature_mul_weight = np.dot(train_features,self.weights)
            train_probab_pred =  self.compute_softmax(feature_mul_weight)
            
            _error = train_targets-train_probab_pred
            #Calculating gradient ascent
            if(boosting == False):
                change_in_weight = self.learning_rate*np.dot(train_features.transpose(),_error)
                self.weights = self.weights+change_in_weight
            elif(boosting == True):
                for i in range(_error.shape[1]):
                    _error[:,i] = _error[:,i]*sample_weight
                change_in_weight = self.learning_rate*np.dot(train_features.transpose(),(_error))
                self.weights = self.weights+change_in_weight
    
    def compute_softmax(self,_weighted_feature_array,_axis=1):
        #Used the function from project 1 run_3a_b_c.py
        """
            This function computes the softmax probablities as per the softmax formula involving exponenets.
        """
        exponent_weighted_feature_array = np.exp(_weighted_feature_array - np.max(_weighted_feature_array,axis=_axis,keepdims=True)) 
        return exponent_weighted_feature_array/np.sum(exponent_weighted_feature_array,axis=_axis,keepdims=True)

    def softmax_predict(self,feature_array,_axis=1):
        #Used the function from project 1 run_3a_b_c.py
        #predicting probbality for classes when input is given and getting the max probability index.
        feature_dot_weight = np.dot(feature_array,self.weights)
        return np.argmax(self.compute_softmax(feature_dot_weight,_axis=_axis),axis=_axis)
    
    def softmax(self,feature_array,_axis=1):
        #Used the function from project 1 run_3a_b_c.py
        #predicting probbality for classes when input is given
        feature_dot_weight = np.dot(feature_array,self.weights)
        return self.compute_softmax(feature_dot_weight,_axis=_axis)


#1    
def bagging(X_train, Y_train, X_test, Y_test, num_regressors,unique_classes):
    _regressor = []
    np.random.seed(5+num_regressors)
    for i in range(num_regressors):
        model = softmax_regression()
        sample_ids = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_sample = X_train[sample_ids]
        Y_sample = Y_train[sample_ids]
        model.softmax_regression(X_sample,Y_sample,unique_classes)
        _regressor.append(model)
    predictions = np.zeros((len(X_test), len(_regressor))).astype(int)
    for i, model in enumerate(_regressor):
        predictions[:, i] = model.softmax_predict(X_test)
    Y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
    
    return np.sum(Y_pred == np.argmax(Y_test,axis=1)) / len(np.argmax(Y_test,axis=1))

#2
def adaboost(X_train, Y_train, X_test, Y_test, num_estimators,unique_classes,learn_rate,epochs):
    #Intializing sample weights
    weights = np.ones(X_train.shape[0])/X_train.shape[0]
    final_prediction_test = np.zeros((X_test.shape[0],len(unique_classes)))
    _estimator_model = []
    
    for i in range(num_estimators):
        #Training the classifier on sample weights
        model = softmax_regression(epochs,learn_rate)
        model.softmax_regression(X_train,Y_train,unique_classes,boosting = True,sample_weight = np.array(weights))
        
        # Computing error and error_rate and updating sample weights
        Y_pred = model.softmax_predict(X_train)
        incorrect = (Y_pred != np.argmax(Y_train,axis=1))
        error = np.sum(weights[incorrect])/np.sum(weights)
        error_rate = 0.5*np.log((1-error)/error) + np.log(len(unique_classes)-1)
        weights = weights*np.exp(error_rate*incorrect)
        weights = weights/np.sum(weights)
        _estimator_model.append((model,error_rate))
    
    # Predictions on test data
    for model, error_rate in _estimator_model:
        _prediction =  model.softmax(X_test)
        final_prediction_test += error_rate*_prediction
    final_prediction_test = np.argmax(final_prediction_test,axis=1)
    
    return np.mean(final_prediction_test == np.argmax(Y_test,axis=1))




#3
############################### K-means ###############################################################
def load_data_kmeans(train_data_path,remove_from_end=0):
    """
        This is a function to load the training data for classification problems required for kmeans.
        This returns the feature array and target array with index for each label.
    """
    _feat = []
    _target = []
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-1-remove_from_end)]])
            _target.append(temp_data[len(temp_data)-1])    
        
        unique_labels = list(set(_target))
        target_array = np.array(_target)
        single_hot_array = np.zeros((len(_target),len(unique_labels)))
        for _index,_val in enumerate(unique_labels):
            single_hot_array[:,_index] =  (target_array == _val).astype(int)
    
    return np.array(_feat),np.argmax(single_hot_array,axis=1)


#REF: https://anderfernandez.com/en/blog/kmeans-algorithm-python/
class KMeans:
    def __init__(self, features, k, epochs=100):
        self.k_value = k
        self.features = features
        self.epoch = epochs

    def train(self):
        #This function is used to train the kmeans model.
        self.clusters_init()
        for i in range(self.epoch):
            self.cluster_modification()
            self.update_clusters()

    def clusters_init(self):
        #This function randomly creates clusters for intialization.
        self.clusters = self.features[:self.k_value]
        
    def cluster_modification(self):
        self.cluster_labels = np.argmin(self.cartesian_distance(), axis=0)

    def update_clusters(self):
        #After cluster is modified in each iteration the labels are updated for each data point.
        clusters = []
        for _k in range(self.k_value):
            clusters.append(self.features[self.cluster_labels == _k].mean(axis=0))
        self.clusters = np.array(clusters)
        
    def cartesian_distance(self):
        return np.sqrt(((self.features - self.clusters[:, None])**2).sum(axis=2))

    def accuracy(self, targets):
        ## This function calculates each cluster accuracy and return overall accuracy.
        accuracies = []
        for _k in range(self.k_value):
            _data = [targets[np.where(self.cluster_labels == _k)]]
            frequency_of_data_in_clusters = np.unique(_data, return_counts=True)
            most_frequent_label = frequency_of_data_in_clusters[0][np.argmax(frequency_of_data_in_clusters[1])]
            accuracy = (frequency_of_data_in_clusters[1][most_frequent_label]/sum(frequency_of_data_in_clusters[1]))
            print("Cluster no = ",_k," most frequent label = ",most_frequent_label," acuuracy = ",accuracy)
            fractional_weights = ((self.features[self.cluster_labels == _k]).shape[0])/self.features.shape[0]
            accuracies.append(fractional_weights*accuracy)
        return sum(accuracies)
