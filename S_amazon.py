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

def read_data_amazon(filename):
    df = pd.read_csv(filename,
                   sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot_table(index='user', columns='item', values='rate')
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
    ############P QB 或 P Q B
    p.to_csv('P_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')
    q.to_csv('Q_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')
    B_baseline.to_csv('B_'+file_name+'.csv', sep='\n', index=False, header=False, float_format='%.5f')

def save_rating(file_name, mf):
    matrix = mf.full_matrix()
    np.save('amazon-result-test/'+file_name, matrix)

def auto_read(data_path, namelist):
    all_r = []
    for item in namelist:
        R = read_data_amazon(data_path + item + ".csv")
        all_r.append(R)
    return all_r

data_path = os.getcwd() + '/amazon/'

first = read_data_amazon(data_path + 'amazon_mon1.csv')
print(first.shape) #####去掉.csv 在下面加
namelist_amazon = ['amazon_mon2', 'amazon_mon3', 'amazon_mon4', 'amazon_mon5', 'amazon_mon6', 'amazon_mon7', 'amazon_mon8', 'amazon_mon9', 'amazon_mon10', 'amazon_mon11', 'amazon_mon12', 'amazon_mon13', 'amazon_mon14', 'amazon_mon15']

R = auto_read(data_path, namelist_amazon)
start = time.time()
mf_b = MF_b(first, K=300, alpha=0.01, beta=0.001, iterations=1000)
training_process = mf_b.train()
save_data('amazon_mon1', mf_b)
save_rating('amazon_mon1', mf_b)
############single_train
for index, item in enumerate(namelist_amazon):
    mf_b.singletrain(R[index], 100)
    save_rating(item, mf_b)
    save_data(item, mf_b)
############ori_train
# for index, item in enumerate(namelist_amazon):
#     mf_b = MF_b(R[index], K=300, alpha=0.01, beta=0.001, iterations=500)
#     training_process = mf_b.train()
#     save_rating(item, mf_b)