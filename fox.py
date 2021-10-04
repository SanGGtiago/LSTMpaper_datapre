import sys
import os
sys.path.append(os.getcwd())

from MF_b_improve import MF_b

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
import os
import operator
import time
from functools import reduce
import contextlib

def read_data_fox(filename):
    df = pd.read_csv(filename,
                   sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot(index='user', columns='item', values='rate')
    matrix.fillna(0, inplace=True)

    matrix_list = matrix.iloc[:,:].values

    return matrix_list

def save_data(file_name, mf):
    P = mf.P.flatten()
    p = pd.DataFrame([P]) 

    Q = mf.Q.T.flatten()
    q = pd.DataFrame([Q]) 

    B_u = mf.b_u.flatten()
    bu_ = pd.DataFrame([B_u])
    B_i = mf.b_i.flatten()
    bi_ = pd.DataFrame([B_i])
    B_ = mf.b
    b_ = pd.DataFrame([B_]) 

    #PQ = pd.concat([p, q], axis = 1)
    B_baseline = pd.concat([bu_, bi_, b_], axis = 1)
    p.to_csv('Mfbfox_P_'+file_name, sep='\n', index=False, header=False, float_format='%.5f')
    q.to_csv('Mfbfox_Q_'+file_name, sep='\n', index=False, header=False, float_format='%.5f')
    B_baseline.to_csv('Mfbfox_B_'+file_name, sep='\n', index=False, header=False, float_format='%.5f')

def auto_read(filepath, name):
    all_r = []
    for item in name:
        R = read_data_fox(filepath + item)
        all_r.append(R)

    return all_r

data_path = os.getcwd() + '/Data/fox/'

#data_path = os.getcwd() + '/Data/100k
# /accumulate/'

first = read_data_fox(data_path + 'mon0.csv')

namelist_fox = ['mon1.csv', 'mon2.csv', 'mon3.csv', 'mon4.csv', 'mon5.csv', 'mon6.csv', 'mon7.csv', 'mon8.csv']

R = auto_read(data_path, namelist_fox)
start = time.time()
mf_b = MF_b(first, K=300, alpha=0.01, beta=0.001, iterations=500)
training_process = mf_b.train()
save_data('mon1.csv', mf_b)

for index, item in enumerate(namelist_fox):
    mf_b.singletrain(R[index], 100)
    save_data(item, mf_b)
