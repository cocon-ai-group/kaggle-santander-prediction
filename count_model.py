import os
import sys
import time
import pickle
import itertools
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import Process
from collections import Counter


parser = argparse.ArgumentParser(description='Santander Value Prediction Challenge Regression Kernel')
parser.add_argument('--num_process', '-p', type=int, default=8,
					help='use cpus')
parser.add_argument('models', metavar='N', nargs='+',
					help='model file for the accumulator')
args = parser.parse_args()

print('read data:')

df = pd.read_csv('../input/train.csv')
cols = [f for f in df.columns if f not in ["ID", "target"]]
print('%d cols read.'%len(cols))

nlet_model = []
for model in args.models:
	with open(model, mode='rb') as f:
		rules = pickle.load(f)
		nlet_model.extend(rules)

print('%d rules read:'%len(nlet_model))

print('count train target:')
train_val = np.zeros((df.ID.values.shape[0],5))

def get_train_res(idx):
	dd = df.loc[idx]
	target_lst = []
	for tr in nlet_model:
		n_nlet = (len(tr)-2) // 2
		col1 = tr[0:n_nlet]
		col2 = tr[n_nlet:n_nlet*2]
		col3 = [tr[n_nlet*2]]
		src = dd[col2+['ID']].values
		if not 0 in src[0:n_nlet]:
			for tgt in df[col1+['ID']+col3].values:
				if (src[0:n_nlet]==tgt[0:n_nlet]).all() and src[n_nlet]!=tgt[n_nlet]:
					target_lst.append(tgt[n_nlet+1])
	return idx, target_lst

proc_pool = Pool(args.num_process)

result = proc_pool.map(get_train_res, df.index)
for i, w in result:
	train_val[i][0] = 0 if len(w)==0 else np.mean(w)
	train_val[i][1] = 0 if len(w)==0 else np.max(w)
	train_val[i][2] = 0 if len(w)==0 else np.median(w)
	train_val[i][3] = 0 if len(w)==0 else np.std(w)
	train_val[i][4] = len(w)

df_d = pd.DataFrame(train_val, columns=['mean', 'max', 'median', 'std', 'len'])
df_d['ID'] = df['ID']
df_d['target'] = df['target']

print('%d/%d columns filled in train:'%(len(df_d[df_d['mean']!=0]),len(df_d)))
print('mean/max rules in matchs is (%f/%d).'%(df[df.len!=0].mean().len,df[df.len!=0].max().len))



