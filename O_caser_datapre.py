import pandas as pd
import glob

def convert():
    dataset = 'fox'
    file_path = glob.glob(f'{dataset}/'+'*.csv')
    for i, file_name in enumerate(file_path):
        df = pd.read_csv(file_name,
                    sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
        df = df.drop(columns=['time'])
        df.to_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=' ', index=False, header=False, float_format='%.5f')

def del_same():
    dataset = '100k'
    i = 3
    j = 8
    F = pd.read_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    S = pd.read_csv(f'O-caser-dataset/{dataset}_mon{j}.txt', sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    df = pd.concat([F, S])
    df = df.drop_duplicates(keep=False)
    df.to_csv(f'O-caser-dataset/{dataset}_test.txt', sep=' ', index=False, header=False, float_format='%.5f')

del_same()