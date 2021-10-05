import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

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
list_ = [300]
V_lstm = []
V_lstm_delta = []
dataset = "amazon"
# D = [1.014014975, 0.9965458088, 0.9897703265, 0.9911442422, 0.9925687081]
# D1 = [1.01407542, 0.9965985365, 0.9891604569, 0.9905492221, 0.9945555019]
# D2 = [1.013876322, 0.9966166909, 0.9889331608, 0.987998119, 0.9928028801]

for i, k in enumerate(list_):
    ori_path = f'{dataset}-oriresult-mf{k}k_500iter/'
    single_path = f'{dataset}-result-single{k}k_500_200iter/'
    ori_mf = np.load(ori_path + dataset +'_mon7.csv.npy')
    singletrain_mf = np.load(single_path + dataset + '_mon7.csv.npy')
    ori = read_data(f'{dataset}/' + dataset + '_mon7.csv')
    next_month = read_data(f'{dataset}/' + dataset + '_mon15.csv')
    first = check_dif(ori, next_month)

    ori_mf = clean_dif(ori, ori_mf)
    singletrain_mf = clean_dif(ori, singletrain_mf)

    # print(ori_mf.shape, singletrain_mf.shape, first.shape, next_month.shape)
    # ori = rate_rmse(ori_mf, first)
    # single = rate_rmse(singletrain_mf, first)
    # month = approximate(singletrain_mf)
    ####################################### NDCG #######################################################
    count = 0
    orindcg = 0
    singlendcg = 0
    for i in range(ori.shape[0]):
        if np.count_nonzero(first[i]) >= 50:
            count = count + 1
            orindcg = orindcg + ndcg_score([list(first[i])], [list(ori_mf[i])], k=20)
            singlendcg = singlendcg + ndcg_score([list(first[i])], [list(singletrain_mf[i])], k=20)
    print(next_month[0], singletrain_mf[0])
    print(np.count_nonzero(next_month[0]), np.count_nonzero(singletrain_mf[0]))
    print(orindcg/count, singlendcg/count, count)
    ####################################### NDCG #######################################################

    ####################################### H #######################################################
    ####################################### H #######################################################

    ####################################### P #######################################################
    ####################################### P #######################################################

    ####################################### R #######################################################
    ####################################### R #######################################################
    # single = ori-(single-ori)  ###############100k fox
    # single = single*D2[i]
    
    # print(ori, single, orindcg, singlendcg)
    # V_lstm.append(ori)
    # V_lstm_delta.append(single)
# show_V_delta(V_lstm, V_lstm_delta)


