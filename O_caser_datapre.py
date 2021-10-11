import pandas as pd
import glob

dataset = ''

def convert():
    file_path = glob.glob(f'{dataset}/'+'*.csv')
    for i, file_name in enumerate(file_path):
        df = pd.read_csv(file_name,
                    sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
        df = df.drop(columns=['time'])
        df.to_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=' ', index=False, header=False, float_format='%.5f')

def del_same_between2mon(i, j):
    F = pd.read_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=' ', names=['user','item','rate'], engine='python', encoding='latin-1')
    S = pd.read_csv(f'O-caser-dataset/{dataset}_mon{j}.txt', sep=' ', names=['user','item','rate'], engine='python', encoding='latin-1')
    df = pd.concat([F, S])
    df = df.drop_duplicates(subset=['user', 'item']).reset_index(drop=True) 
    df.to_csv(f'O-caser-dataset/{dataset}_test.txt', sep=' ', index=False, header=False, float_format='%.5f')

def del_same_rating(i):
    df = pd.read_csv(f'O-caser-dataset/{dataset}_mon{i}.txt', sep=' ', names=['user','item','rate'], engine='python', encoding='latin-1')
    df = df.drop_duplicates(subset=['user', 'item']).reset_index(drop=True) 
    df.to_csv(f'O-caser-dataset/{dataset}_test10.txt', sep=' ', index=False, header=False, float_format='%.5f')

# del_same_rating(10)
# del_same_between2mon(10,15)