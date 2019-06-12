# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:18:22 2019

@author: 簡佑穎
"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import time

features = []
labels = []
with open('validation_70%_111Var.txt') as file:
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
with open('testing_30%_111Var.txt') as file:
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

def XGBoost(X_train, y_train):
    xgb = XGBClassifier(max_depth=dep, 
                        learning_rate=lr, 
                        n_estimators=100, 
                        silent=True, 
                        booster='gbtree', 
                        gamma=gam, 
                        min
                        )


    xgb.fit(X_train, y_train)

    return xgb
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
gamma = [0.03, 0.052, 0.02, 0.014, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
learning_rate=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for depth in range(1,11,1):
    for n in range(5,51,5):
        for gam in gamma:
            for lr in learning_rate:
                for train, test in skf.split(features,labels):
                    X_train, X_test = features[train], features[test]
                    y_train, y_test = labels[train], labels[test]
                                
                    algo_list = [XGBoost(X_train, y_train)]
                    for algo in algo_list:
                        all_indicator = Indicator(algo, X_test, y_test)
                        cnf = all_indicator['cnf']
                        f1 = all_indicator['f_measure']
                        precision = all_indicator['precision_measure']
                        recall = all_indicator['recall_measure']
                        print('train','\t','10_fold','\t',depth,'\t',n,'\t',lr,'\t',
                              cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                              f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                              file=open('result_10fold_XGB_70vali_111VarTrai.txt', 'a+'))
                        
                    for algo in algo_list:
                        all_indicator = Indicator(algo, test_features, test_labels)
                        cnf = all_indicator['cnf']
                        f1 = all_indicator['f_measure']
                        precision = all_indicator['precision_measure']
                        recall = all_indicator['recall_measure']
                        print('test','\t','10_fold','\t',depth,'\t',n,'\t',lr,'\t',
                              cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                              f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                              file=open('result_10fold_XGB_70vali_111VarTest.txt', 'a+')) 
        

end_time = time.time()
print("Total time taken:",round(end_time-start_time,2),'seconds')