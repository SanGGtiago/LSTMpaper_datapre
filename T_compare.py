import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from collections import defaultdict

def read_data(filename):
    df = pd.read_csv(filename,
                   sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot_table(index='user', columns='item', values='rate')
    matrix.fillna(0, inplace=True)
    matrix_list = matrix.iloc[:,:].values

    return matrix_list

def rate_rmse(predict, ori):
    """
    A function to compute the total mean square error
    """
    xs, ys = ori.nonzero()
    predicted = predict
    error = 0
    for x, y in zip(xs, ys):
        error += pow(ori[x, y] - predicted[x, y], 2)

    return np.sqrt(error/len(xs))

def get_top_n(predictions, k):
    #np.array([9,1,3,4,8,7,2,5,6,0])
    temp = np.argpartition(predictions, -k)[-k:]

    return temp

def check_dif(ori, next_month):

    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            if next_month[i][j] == ori[i][j]:
                next_month[i][j] = 0
    return next_month

def clean_dif(ori, next_month):

    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            if ori[i][j] != 0:
                next_month[i][j] = 0
    return next_month

def approximate(mon):
    for i in range(mon.shape[0]):
        for j in range(mon.shape[1]):
            if mon[i, j]>=5:
                mon[i, j] = 5
            
            # mon[i, j] = round(mon[i, j])

    return mon

def min(ori, _):
    for i in range(len(ori)):
        m = ori[i] - _[i]
        ori[i] = ori[i] - m
    return ori 

    ####################################### NDCG #######################################################
def ndcg(TD, OD, PD, k, num):
    #num是資料裡電影數大於的數目
    count = 0
    orindcg = 0
    singlendcg = 0
    for i in range(TD.shape[0]):
        if np.count_nonzero(TD[i]) >= num:
            count = count + 1
            orindcg = orindcg + ndcg_score(np.asarray([TD[i]]), np.asarray([OD[i]]), k=k)
            singlendcg = singlendcg + ndcg_score(np.asarray([TD[i]]), np.asarray([PD[i]]), k=k)
    print(orindcg/count, singlendcg/count, count)
    return 0
    ####################################### NDCG #######################################################

    ####################################### H #######################################################
def hit(TD, POD, k, num):
    #num是資料裡電影數大於的數目 , 時間複雜度要修改
    hit = 0
    check = 0
    for i in range(TD.shape[0]):
        if np.count_nonzero(TD[i]) >= num:
            T = get_top_n(TD[i], k)
            for item in T:
                error = abs(TD[i, item] - POD[i, item])    
                if error < 0.5:
                    hit+=1
                    check +=1
                else:
                    check +=1
    print(hit/check)
    return 0
    ####################################### H #######################################################

    ####################################### PR #######################################################
def P_R(TD, POD, k, num):
    #num是資料裡電影數大於的數目 , 時間複雜度要修改
    usercount = 0
    TP = 0
    FN = 0
    FP = 0
    check = 0
    for i in range(TD.shape[0]):
        if np.count_nonzero(TD[i]) >= num:
            usercount = usercount + 1
            T = get_top_n(TD[i], k)
            P = get_top_n(POD[i], k)
            for Titem in T:
                for j in range(10):
                    if Titem == P[j]:
                        TP = TP + 1
                        check = 1
                if check == 0: FN = FN +1
                check = 0
            check = 0
            for Pitem in P:
                for j in range(10):
                    if Pitem == T[j]:
                        check = 1
                if check == 0: FP = FP +1
                check = 0
    print(TP, FN, FP)
    print('P:',TP/(TP+FP), '   R:',+TP/(TP+FN))
    return 0
    ####################################### PR #######################################################

list_ = [300]
V_lstm = []
V_lstm_delta = []
dataset = "amazon"

for i, k in enumerate(list_):
    ori_mf = np.load(f'{dataset}-oriresult/' + dataset +'_mon10.npy')
    singletrain_mf = np.load(f'{dataset}-singleresult/' + dataset + '_mon10.npy')
    ori = read_data(f'{dataset}/' + dataset + '_mon10.csv')
    next_month = read_data(f'{dataset}/' + dataset + '_mon15.csv')

    first = check_dif(ori, next_month)
    ori_mf = clean_dif(ori, ori_mf)
    singletrain_mf = clean_dif(ori, singletrain_mf)

    ndcg(first, ori_mf, singletrain_mf, 50, 50)
    # hit(first, ori_mf, 10, 50)
    # hit(first, singletrain_mf, 10, 50)
    # P_R(first, ori_mf, 10, 50)
    # P_R(first, singletrain_mf, 10, 50)



