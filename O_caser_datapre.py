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
    dataset = '1M'
    i = 10
    j = 15
    F = pd.read_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=' ', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    S = pd.read_csv(f'O-caser-dataset/{dataset}_mon{j}.txt', sep=' ', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    df = pd.concat([F, S]).drop_duplicates().reset_index(drop=True)
    df.to_csv(f'O-caser-dataset/{dataset}_test.txt', sep=' ', index=False, header=False, float_format='%.5f')

del_same()