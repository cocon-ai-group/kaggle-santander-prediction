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

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
ID = df_test.ID.values
df_train = df_train.drop(['target'], axis=1)
df = pd.concat([df_test,df_train], axis=0, sort=False, ignore_index=True)

cols = [f for f in df.columns if f not in ["ID", "target"]]
print('%d cols read.'%len(cols))

nlet_model = []
for model in args.models:
	with open(model, mode='rb') as f:
		rules = pickle.load(f)
		nlet_model.extend(rules)

print('%d rules read:'%len(nlet_model))

print('submit model:')
train_val = np.zeros((ID.shape[0],5))

def get_train_res(idx):
	print('%d/%d process...'%(idx,len(ID)))
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

result = proc_pool.map(get_train_res, df.index[0:ID.shape[0]])
for i, w in result:
	train_val[i][0] = 0 if len(w)==0 else np.mean(w)
	train_val[i][1] = 0 if len(w)==0 else np.max(w)
	train_val[i][2] = 0 if len(w)==0 else np.median(w)
	train_val[i][3] = 0 if len(w)==0 else np.std(w)
	train_val[i][4] = len(w)

df_d = pd.DataFrame(train_val, columns=['mean', 'max', 'median', 'std', 'len'])
df_d['ID'] = ID

df_d.to_csv("submit_src.csv")

df_d.rename({'mean':'target'}).drop(['max', 'median', 'std', 'len'], axis=1).to_csv('submit_mean.csv', index=False)
df_d.rename({'max':'target'}).drop(['mean', 'median', 'std', 'len'], axis=1).to_csv('submit_max.csv', index=False)
df_d.rename({'median':'target'}).drop(['mean', 'max', 'std', 'len'], axis=1).to_csv('submit_median.csv', index=False)

print('%d/%d columns filled in submission:'%(len(df_d[df_d['mean']!=0]),len(df_d)))
print('mean/max rules in matchs is (%f/%d).'%(df_d[df_d.len!=0].mean().len,df_d[df_d.len!=0].max().len))
print('mean/max stds in no-allcorrect matchs is (%f/%d).'%(df_d[df_d['std']!=0]['std'].mean(),df_d[df_d['std']!=0]['std'].max()))
print('%d matchs is no-correct std>1.'%(len(df_d[df_d['std']>1.0]['std']),))
print('mean/max stds in no-correct matchs is (%f/%d).'%(df_d[df_d['std']>1.0]['std'].mean(),df_d[df_d['std']>1.0]['std'].max()))



