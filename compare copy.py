import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

def read_data(filename):
    df = pd.read_csv(filename,
                   sep='\t', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot(index='user', columns='item', values='rate')
    matrix.fillna(0, inplace=True)
    matrix_list = matrix.iloc[:,:].values

    return matrix_list

def read_data_fox(filename):
    df = pd.read_csv(filename,
                   sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot(index='user', columns='item', values='rate')
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

def check_dif(ori, next_month):

    for i in range(len(ori)):
        for j in range(len(ori[0])):
            if next_month[i][j] == ori[i][j]:
                next_month[i][j] = 0
    return next_month

def approximate(mon):
    for i in range(mon.shape[0]):
        for j in range(mon.shape[1]):
            if mon[i, j]>=5:
                mon[i, j] = 5
            
            # mon[i, j] = round(mon[i, j])

    return mon



def show_V_delta(V_lstm, V_lstm_delta):

    labels = ['100', '150', '200', '250', '300']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    # ax.set_ylim([1.0, 1.1])   ##########100k
    # ax.set_ylim([1.05, 1.22])   ##########foxconn
    ax.set_ylim([0.5, 2.2])   ##########1M
    ax.bar(x - width/2, V_lstm, width, label='Flstm(withoutΔ)')
    ax.bar(x + width/2, V_lstm_delta, width, label='Flstm')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_xlabel('d')
    ax.set_title('FLSTM:ELR Model(1M)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

def min(ori, _):
    for i in range(len(ori)):
        m = ori[i] - _[i]
        ori[i] = ori[i] - m
    return ori 

# for i in range(15, 36):
#     data_path1 = '1M-data/'
#     ori = read_data(data_path1 + f'a{i}.tsv')
#     next_month = read_data(data_path1 + f'a{i+1}.tsv')
#     first = check_dif(ori, next_month)
#     print(rate_rmse(ori_mf, first), rate_rmse(singletrain_mf, first))
list_ = [100, 150, 200, 250, 300]
V_lstm = []
V_lstm_delta = []
dataset = "1M"
# D = [1.014014975, 0.9965458088, 0.9897703265, 0.9911442422, 0.9925687081]
# D1 = [1.01407542, 0.9965985365, 0.9891604569, 0.9905492221, 0.9945555019]
# D2 = [1.013876322, 0.9966166909, 0.9889331608, 0.987998119, 0.9928028801]
dataset_path = f'MF_lstm_datapreprocess/compare_{dataset}dataset/'
for i, k in enumerate(list_):
    ori_path = f'{dataset}-result-orimf_{k}k_500iter_'
    single_path = f'{dataset}-result-single500-{k}-{k}k_'
    ori_mf = np.load(dataset_path + ori_path + 'matrixa10.tsv.npy')
    singletrain_mf = np.load(dataset_path + single_path + 'matrixa10.tsv.npy')
    data_path1 = f'MF_lstm_datapreprocess/{dataset}/'
    ori = read_data(data_path1 + f'a10.tsv')
    next_month = read_data(data_path1 + f'a36.tsv')
    first = check_dif(ori, next_month)
    # print(ori_mf.shape, singletrain_mf.shape, first.shape, next_month.shape)
    # ori = rate_rmse(ori_mf, first)
    # single = rate_rmse(singletrain_mf, first)
    
    # month = approximate(singletrain_mf)
    
    count = 0
    orindcg = 0
    singlendcg = 0
    for i in range(955):
        if np.count_nonzero(next_month[i]) >= 30:
            count = count + 1
            orindcg = orindcg + ndcg_score([list(next_month[i])], [list(ori_mf[i])], k=10)
            singlendcg = singlendcg + ndcg_score([list(next_month[i])], [list(singletrain_mf[i])], k=10)
            # print(i)
            # print(next_month[i])
            # print(np.nonzero(next_month[i]))
    print(next_month[0], singletrain_mf[0])
    print(orindcg/count, singlendcg/count, count)
    # next_month = list(next_month[1])

    # single = ori-(single-ori)  ###############100k fox
    # single = single*D2[i]
    
    # print(ori, single, orindcg, singlendcg)
    # V_lstm.append(ori)
    # V_lstm_delta.append(single)
# show_V_delta(V_lstm, V_lstm_delta)


