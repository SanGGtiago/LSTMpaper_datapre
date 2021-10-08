import sys
import os
sys.path.append(os.getcwd())

from MF_b_improve import MF_b

import glob
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

def read_data(filename):
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
    p.to_csv(dataset+f'-{mode}result/P_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')
    q.to_csv(dataset+f'-{mode}result/Q_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')
    B_baseline.to_csv(dataset+f'-{mode}result/B_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')

def save_rating(file_name, mf):
    matrix = mf.full_matrix()
    np.save(dataset+f'-{mode}result/'+file_name, matrix)
def auto_read(filepath, name):
    all_r = []
    for item in name:
        R = read_data(filepath + item)
        all_r.append(R)

    return all_r


dataset = '100k'
mode = 'ori' # model = ''

file_path = glob.glob(f'{dataset}/*.csv')
first = read_data(file_path[0])
start = time.time()
mf_b = MF_b(first, K=300, alpha=0.01, beta=0.001, iterations=1000)
training_process = mf_b.train()

save_data(dataset+'_mon1', mf_b)
save_rating(dataset+'_mon1', mf_b)
############single_train
# for i, file_name in enumerate(file_path[1:]):
#     data = read_data(file_name)
#     mf_b.singletrain(data, 150)
#     save_rating(f'{dataset}_mon{i}', mf_b)
#     save_data(f'{dataset}_mon{i}', mf_b)

############ori_train
for i, file_name in enumerate(file_path[1:]):
    data = read_data(file_name)
    mf_b = MF_b(data, K=300, alpha=0.01, beta=0.001, iterations=500)
    training_process = mf_b.train()
    save_rating(f'{dataset}_mon{i}', mf_b)
    save_data(f'{dataset}_mon{i}', mf_b)

###################################################
