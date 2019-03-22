# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:24:57 2019

@author: user
"""

f = open('CPA.txt')
name = []
time = []
result = []
for x in f.readline().strip().split('\t'):
    name.append(x)
#print(len(name))
#print(name)

i = 0    
for i in range(len(name)):
    t = name.count(name[i])
    time.append(str(t))
    
print(time)


with open('result.txt', 'w+') as f1:
    f1.write('\n'.join(time))
