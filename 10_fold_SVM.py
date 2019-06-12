# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:01:40 2019

@author: 簡佑穎
"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import time

features = []
labels = []
with open('validation_70%_72Var.txt') as file:
    all_features = file.readlines()
    for i in range(len(all_features)):
        features.append((all_features[i].split('\t')[1:]))
        labels.append((all_features[i].split('\t')[0]))
    features = sp.array(features)
    labels = sp.array(labels)

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
    
test_features = []
test_labels = []
with open('testing_30%_72Var.txt') as file:
    all_features_test = file.readlines()
    for i in range(len(all_features_test)):
        test_features.append((all_features_test[i].split('\t')[1:]))
        test_labels.append((all_features_test[i].split('\t')[0]))
    test_features = sp.array(test_features)
    test_labels = sp.array(test_labels)

scaler = StandardScaler()
scaler.fit(test_features)
test_features = scaler.transform(test_features)

start_time = time.time()

def SVM(X_train, y_train):
    svm = SVC(C=c, gamma=gam)
    svm.fit(X_train, y_train)
    return svm

# 評估模型指標-------------------------------------------------
def Indicator(algo, X, y):
    if 'keras' in str(algo):
        y_pred = algo.predict_classes(X_test)
    else:
        y_pred = algo.predict(X)
    cnf = confusion_matrix(y, y_pred)
    f_measure = f1_score(y, y_pred, average='weighted') # F1-score
    precision_measure = precision_score(y, y_pred, average=None)
    recall_measure = recall_score(y, y_pred, average=None)
    all_indicator = {'cnf': cnf,
                     'f_measure':f_measure,
                     'precision_measure':precision_measure,
                     'recall_measure':recall_measure
                     }
    return all_indicator



skf = StratifiedKFold(n_splits = 10, random_state = 0)
#C = [1, 10, 100, 1000, 10000, 100000, 500000, 1000000, 5000000, 8000000, 9000000, 10000000, 20000000, 30000000, 40000000, 50000000]
#gamma = [0.03, 0.052, 0.02, 0.014, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

gamma = [0.00005,0.00004,0.00003,0.00002,0.00001, 
         0.000009,0.000008,0.000007,0.000006,0.000005,0.000004,0.000003,0.000002,0.000001,
         0.0000009,0.0000008,0.0000007,0.0000006,0.0000005,0.0000004,0.0000003,0.0000002,0.0000001]

for c in range(3000000,8000001,100000):
    for gam in gamma:
        for train, test in skf.split(features,labels):
            X_train, X_test = features[train], features[test]
            y_train, y_test = labels[train], labels[test]
            
            algo_list = [SVM(X_train, y_train)]
            for algo in algo_list:
                all_indicator = Indicator(algo, X_test, y_test)
                cnf = all_indicator['cnf']
                f1 = all_indicator['f_measure']
                precision = all_indicator['precision_measure']
                recall = all_indicator['recall_measure']
                print('train','\t','10_fold','\t', c,'\t', gam,'\t',
                      cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                      f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                      file=open('result_10fold_SVM_72CgamTrai.txt', 'a+'))
                
            for algo in algo_list:
                all_indicator = Indicator(algo, test_features, test_labels)
                cnf = all_indicator['cnf']
                f1 = all_indicator['f_measure']
                precision = all_indicator['precision_measure']
                recall = all_indicator['recall_measure']
                print('test','\t','10_fold','\t', c,'\t', gam,'\t',
                      cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                      f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                      file=open('result_10fold_SVM_72CgamTest.txt', 'a+'))        
        

end_time = time.time()
print("Total time taken:",round(end_time-start_time,2),'seconds')