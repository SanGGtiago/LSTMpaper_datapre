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

def recall(predict, ori):
    """
    A function to compute the total mean square error
    """
    xs, ys = ori.nonzero()
    fn = 0
    tp = 0
    for x, y in zip(xs, ys):
        error += pow(ori[x, y] - predict[x, y], 2)

    return np.sqrt(error/len(xs))

def precision(predict, ori):
    """
    A function to compute the total mean square error
    """
    xs, ys = ori.nonzero()
    fp = 0
    tp = 0
    for x, y in zip(xs, ys):
        error += pow(ori[x, y] - predict[x, y], 2)

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

    ####################################### NDCG #######################################################
def ndcg():
    count = 0
    orindcg = 0
    singlendcg = 0
    for i in range(ori.shape[0]):
        if np.count_nonzero(first[i]) >= 50:
            count = count + 1
            orindcg = orindcg + ndcg_score(np.asarray([first[i]]), np.asarray([ori_mf[i]]), k=10)
            singlendcg = singlendcg + ndcg_score(np.asarray([first[i]]), np.asarray([singletrain_mf[i]]), k=10)
    print(orindcg/count, singlendcg/count, count)
    return 0
    ####################################### NDCG #######################################################

    ####################################### H #######################################################
    ####################################### H #######################################################

    ####################################### P #######################################################
    ####################################### P #######################################################

    ####################################### R #######################################################
def rmse():
    ori_rmse = rate_rmse(ori_mf, first)
    single_rmse = rate_rmse(singletrain_mf, first)
    print(ori_rmse, single_rmse)
    return 0
    ####################################### R #######################################################


# for i in range(15, 36):
#     data_path1 = '1M-data/'
#     ori = read_data(data_path1 + f'a{i}.tsv')
#     next_month = read_data(data_path1 + f'a{i+1}.tsv')
#     first = check_dif(ori, next_month)
#     print(rate_rmse(ori_mf, first), rate_rmse(singletrain_mf, first))
list_ = [300]
V_lstm = []
V_lstm_delta = []
dataset = "fox"

for i, k in enumerate(list_):
    ori_mf = np.load(f'{dataset}-oriresult/' + dataset +'_mon3.npy')
    singletrain_mf = np.load(f'{dataset}-singleresult/' + dataset + '_mon3.npy')
    ori = read_data(f'{dataset}/' + dataset + '_mon3.csv')
    next_month = read_data(f'{dataset}/' + dataset + '_mon8.csv')

    first = check_dif(ori, next_month)
    ori_mf = clean_dif(ori, ori_mf)
    singletrain_mf = clean_dif(ori, singletrain_mf)

    rmse()

    # print(ori_mf.shape, singletrain_mf.shape, first.shape, next_month.shape)
    # ori = rate_rmse(ori_mf, first)
    # single = rate_rmse(singletrain_mf, first)
    # month = approximate(singletrain_mf)

    # single = ori-(single-ori)  ###############100k fox
    # single = single*D2[i]
    
    # print(ori, single, orindcg, singlendcg)
    # V_lstm.append(ori)
    # V_lstm_delta.append(single)
# show_V_delta(V_lstm, V_lstm_delta)


