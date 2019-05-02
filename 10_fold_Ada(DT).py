# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:41:30 2019

@author: user
"""
import numpy as np
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
with open('sample.txt') as file:
    all_features = file.readlines()
    for i in range(len(all_features)):
        features.append((all_features[i].split('\t')[1:]))
        labels.append((all_features[i].split('\t')[0]))
    features = sp.array(features)
    labels = sp.array(labels)

start_time = time.time()

def Adaboost(X_train, y_train):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=dep, max_leaf_nodes=nodes,
                                                    class_weight='balanced',random_state=0),
                             algorithm="SAMME", n_estimators=n, learning_rate=0.1,random_state=0)
    ada.fit(X_train, y_train)
    return ada

# 評估模型指標-------------------------------------------------
def Indicator(algo, X_test, y_test):
    if 'keras' in str(algo):
        y_pred = algo.predict_classes(X_test)
    else:
        y_pred = algo.predict(X_test)
    cnf = confusion_matrix(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred, average='weighted') # F1-score
    precision_measure = precision_score(y_test, y_pred, average=None)
    recall_measure = recall_score(y_test, y_pred, average=None)
    all_indicator = {'cnf': cnf,
                     'f_measure':f_measure,
                     'precision_measure':precision_measure,
                     'recall_measure':recall_measure
                     }
    return all_indicator


skf = StratifiedKFold(n_splits = 10, random_state = 0)


for n in range(5,101,5):
    for dep in range(4,51):
        for nodes in range(8,101):
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
                    print(str(algo)[0:3],'\t',n,'\t',dep,'\t',nodes,'\t',
                          cnf[0,0],'\t',cnf[0,1],'\t',cnf[1,0],'\t',cnf[1,1],'\t',
                          f1,'\t',precision[0],'\t',precision[1],'\t',recall[0],'\t', recall[1],'\t',
                          file=open('result_1.txt', 'a+'))

end_time = time.time()
print("Total time taken:",round(end_time-start_time,2),'seconds')