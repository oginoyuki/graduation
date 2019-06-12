# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:41:30 2019

@author: user
"""
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time

features = []
labels = []
with open('14MLTrai.txt') as file:
    all_features = file.readlines()
    for i in range(len(all_features)):
        features.append((all_features[i].split('\t')[1:]))
        labels.append((all_features[i].split('\t')[0]))
    features = sp.array(features)
    labels = sp.array(labels)

#scaler = StandardScaler()
#scaler.fit(features)
#features = scaler.transform(features)
    
test_features = []
test_labels = []
with open('14MLTest.txt') as file:
    all_features_test = file.readlines()
    for i in range(len(all_features_test)):
        test_features.append((all_features_test[i].split('\t')[1:]))
        test_labels.append((all_features_test[i].split('\t')[0]))
    test_features = sp.array(test_features)
    test_labels = sp.array(test_labels)

#scaler = StandardScaler()
#scaler.fit(test_features)
#test_features = scaler.transform(test_features)

start_time = time.time()

def Adaboost(X_train, y_train):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth,
                                                    class_weight='balanced',random_state=0),
                             n_estimators=n, learning_rate=lr,random_state=1)
    ada.fit(X_train, y_train)
    return ada

# 評估模型指標-------------------------------------------------
def Indicator(algo, X, y):
    if 'keras' in str(algo):
        y_pred = algo.predict_classes(X)
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
max_depth=[1, 2]
learning_rate=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
               0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
               0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

for depth in max_depth:
    for n in range(1,51,1):
        for lr in learning_rate:
            for train, test in skf.split(features,labels):
                X_train, X_test = features[train], features[test]
                y_train, y_test = labels[train], labels[test]
                            
                algo_list = [Adaboost(X_train, y_train)]
                for algo in algo_list:
                    all_indicator = Indicator(algo, X_test, y_test)
                    cnf = all_indicator['cnf']
                    f1 = all_indicator['f_measure']
                    precision = all_indicator['precision_measure']
                    recall = all_indicator['recall_measure']
                    print('train','\t','10_fold','\t',depth,'\t',n,'\t',lr,'\t',
                          cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                          f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                          file=open('result_10fold_Ada_14deplrTrai.txt', 'a+'))
                    
                for algo in algo_list:
                    all_indicator = Indicator(algo, test_features, test_labels)
                    cnf = all_indicator['cnf']
                    f1 = all_indicator['f_measure']
                    precision = all_indicator['precision_measure']
                    recall = all_indicator['recall_measure']
                    print('test','\t','10_fold','\t',depth,'\t',n,'\t',lr,'\t',
                          cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                          f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                          file=open('result_10fold_Ada_14deplrTest.txt', 'a+')) 

end_time = time.time()
print("Total time taken:",round(end_time-start_time,2),'seconds')