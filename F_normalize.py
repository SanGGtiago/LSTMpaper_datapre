import pandas as pd
import numpy as np
import glob

dataset = '100k'
file_path = glob.glob(f'{dataset}/'+'*.tsv')
for i, file_name in enumerate(file_path):
    df = pd.read_csv(file_name,
                   sep='	', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    df.to_csv(f'{dataset}/{dataset}_mon{i}.csv', sep=',', index=False, header=False, float_format='%.5f')