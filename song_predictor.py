#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Song Year Predictor

Created on Thu Nov 30 11:04:58 2017

@author: apoorv
"""

import scipy.io as spio
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR


mat = spio.loadmat('/Users/apoorv/MyDocs/CU Courses/Fall 2017/ML/HW/HW4/MSdata.mat')

full_data = mat['trainx']
full_label = mat['trainy']

test_data = mat['testx']

'''
train_data = full_data[0:400000,:]
train_label = full_label[0:400000,:]

validation_data = full_data[400001:,:]
validation_label = full_label[400001:,:]
'''

train_data = full_data
train_label = full_label

scaled_trainData = preprocessing.StandardScaler().fit(train_data)
transform_trainData = scaled_trainData.transform(train_data)
#transform_trainData =  PCA().fit_transform(transform_trainData_scaler)
#scaled_trainData = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
print(len(transform_trainData))
#scaled_trainData = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
#transform_trainData = scaled_trainData.transform(train_data)
'''
scaled_validData = preprocessing.StandardScaler().fit(validation_data)
transform_validation_data = scaled_validData.transform(validation_data)
print(len(transform_validation_data))
'''

scaled_validData = preprocessing.StandardScaler().fit(test_data)
transform_validation_data = scaled_validData.transform(test_data)
#transform_validation_data = PCA().fit_transform(transform_validation_data_scaler)
print(len(transform_validation_data))

reg = MLPRegressor(activation= 'logistic', learning_rate='adaptive', hidden_layer_sizes=(100,), alpha = 0.001, verbose = True, random_state=10,max_iter=400)
#reg = AdaBoostRegressor()
#reg = SVR()
reg.fit(transform_trainData, train_label)

predictions = reg.predict(transform_validation_data)
predictions = np.array(predictions)
predictions = [int(np.ceil(x)) for x in predictions]

#validation_label = np.array(validation_label)

'''
print(predictions)
print(validation_label)
print(len(validation_label))
print(len(predictions))
print(reg.score(predictions, np.asarray(validation_label)))

absolute_error = 0
for i,j in zip(predictions,validation_label):
    absolute_error = absolute_error + np.abs(i-j)
    print(i)
    print(j)
    
mean_absolute_error = absolute_error/validation_label.size
print(mean_absolute_error)
'''


csvFile = open('/Users/apoorv/MyDocs/CU Courses/Fall 2017/ML/HW/HW4/results_scaler_logistic.csv', 'w') 
topRow = "dataid,prediction\n"
csvFile.write(topRow)

i=1
for x in predictions:
    row = str(i) + ',' + str(x) + '\n'
    #print(row)
    csvFile.write(row)
    i = i+1
    
csvFile.close()
   