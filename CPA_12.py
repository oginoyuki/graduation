# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 00:24:52 2019

@author: user
"""

f = open('CPA_12.txt', encoding = 'utf8')
name = []
time = []
result = []

for x in f.readline().strip().split('\t'):
    name.append(x)
#print(len(name))
#print(name[13390])

i = 0    
for i in range(len(name)):
    t = name.count(name[i])
    time.append(str(t))
    
#print(time[26423])


with open('result_12.txt', 'w+') as f1:
    f1.write('\n'.join(time))