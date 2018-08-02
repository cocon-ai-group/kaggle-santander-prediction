import os
import gc
import sys
import time
import pickle
import itertools
import argparse
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import Process
from collections import Counter

parser = argparse.ArgumentParser(description='Santander Value Prediction Challenge Regression Kernel')
parser.add_argument('--num_process', '-p', type=int, default=80,
					help='use cpus')
parser.add_argument('--rarity', '-r', type=int, default=3,
					help='rarity of value')
parser.add_argument('--rarity_bin', '-a', type=int, default=-1,
					help='rarity filter in find balue(default=rarity+2)')
parser.add_argument('--nlet', '-n', type=int, default=3,
					help='n values matching')
parser.add_argument('--rarity_filter', '-i', type=int, default=-1,
					help='use rarity filter in find pair')
parser.add_argument('--bin_size', '-s', type=int, default=10000,
					help='max size of pairs bin')
parser.add_argument('--bag_size', '-b', type=int, default=3000000,
					help='max size of pairs bag')
parser.add_argument('--filter_size', '-f', type=int, default=50,
					help='model filter size')
parser.add_argument('--model', '-m', default='model.pickle',
						help='model output filename')
args = parser.parse_args()
if args.rarity_bin < 0:
	args.rarity_bin = args.rarity+2

print('read data:')

df = pd.read_csv('../input/train.csv')
cols = [f for f in df.columns if f not in ["ID", "target"]]
cnt = Counter(df[cols].values.flatten())
print('%d cols read.'%len(cols))

print('find rera items include rare values:')

def get_rera_index(p):
	if cnt[p] <= args.rarity_bin:
		q = list(set(df[df.values==p].index))
		l = len(q)
		if 1 < l and l <= args.rarity:
			return True, q
	return False, []

proc_pool = Pool(args.num_process)

rera_index = []
result = proc_pool.map(get_rera_index, cnt.keys())
for i, w in result:
	if i:
		rera_index.extend(w)
rera_index = list(set(rera_index))

print('%d items found.'%len(rera_index))

pair_index = itertools.combinations(rera_index, 2)

print('find source and target item in pair:')

def get_find_pair(fi):
	index1 = fi[0]
	index2 = fi[1]
	target1 = df.iloc[index1].target
	target2 = df.iloc[index2].target
	item1 = (df.iloc[index1])[df.iloc[index1].values==target2].index
	item2 = (df.iloc[index2])[df.iloc[index2].values==target1].index
	len1 = len(item1)
	len2 = len(item2)
	if len1!=0 and len2==0:
		return True, [(index1,index2,target2)]
	elif len2!=0 and len1==0:
		return True, [(index2,index1,target1)]
	elif len2!=0 and len1!=0:
		return True, [(index1,index2,target2),(index2,index1,target1)]
	else:
		return False, []

proc_pool = Pool(args.num_process)

find_pair = []
result = proc_pool.map(get_find_pair, pair_index)
for i, w in result:
	if i:
		find_pair.extend(w)

print('%d pairs found.'%len(find_pair))

print('make nlet columns that maching in pair:')

def get_nlet_target(pr):
	result_target = []
	index1 = pr[0]
	index2 = pr[1]
	target = pr[2]
	find_vals = [c for c in df[cols].iloc[index1].values if c != 0]
	find_cols = []
	if args.rarity_filter > 0:
		find_vals = [c for c in find_vals if cnt[c] <= args.rarity_filter]
	for f in find_vals+[target]:
		find_cols.extend(list((df.iloc[index1])[df.iloc[index1].values==f].index))
		find_cols.extend(list((df.iloc[index2])[df.iloc[index2].values==f].index))
	find_cols = list(set(find_cols))
	find_cols.remove('target')
	df_pair = df[find_cols].iloc[[index1,index2]]
	iter_pairs = []
	for _c in df_pair.columns:
		for _d in df_pair.columns:
			if df_pair[_d].loc[index1] != 0 and df_pair[_d].loc[index1] == df_pair[_c].loc[index2]:
				iter_pairs.append((_d,_c))
	tgt_idx = (df.iloc[index1])[df.iloc[index1].values==target].index
	if len(iter_pairs) >= args.nlet:
		for iter in itertools.combinations(iter_pairs, args.nlet):
			for tgt in tgt_idx:
				lst1 = [iter[n][0] for n in range(args.nlet)]
				lst2 = [iter[n][1] for n in range(args.nlet)]
				if len(set(lst1)) == len(lst1) and len(set(lst2)) == len(lst2):
					lst = lst1 + lst2 + [tgt]
					result_target.append(lst)
				if len(result_target) >= args.bin_size:
					return result_target
	return result_target

proc_pool = Pool(args.num_process)

nlet_target = []
result = proc_pool.map(get_nlet_target, find_pair)
for w in result:
	nlet_target.extend(w)

print('%d nlet pairs made.'%len(nlet_target))

if 0 < args.bag_size and args.bag_size < len(nlet_target):
	nlet_target = random.sample(nlet_target, args.bag_size)
	print('%d nlet pairs in bag.'%len(nlet_target))

print('caluclate each nlet columns:')

def get_nlet_model(tr):
	num = 0
	col1 = [tr[n] for n in range(args.nlet)]
	col2 = [tr[n+args.nlet] for n in range(args.nlet)]
	col3 = [tr[args.nlet*2]]
	for ia in df[col1+['ID']+col3].values:
		if not 0 in ia[0:args.nlet]:
			for ib in df[col2+['ID','target']].values:
				if (ib[0:args.nlet]==ia[0:args.nlet]).all() and ia[args.nlet]!=ib[args.nlet]:
					num += 1
					if ia[args.nlet+1] != ib[args.nlet+1]:
						return False, (0,)
	if num > args.filter_size:
		print('if column (%s) eq (%s) then target is %s. in %d items.'%(' '.join(col1),' '.join(col2),col3[0],num))
		return True, tr+[num]
	else:
		return False, []

proc_pool = Pool(args.num_process)

nlet_model = []
result = proc_pool.map(get_nlet_model, nlet_target)
for i, w in result:
	if i:
		nlet_model.append(w)

print('%d models built.'%len(nlet_model))

del cols, cnt, rera_index, pair_index, find_pair, nlet_target
gc.collect()

with open(args.model, mode='wb') as f:
	pickle.dump(nlet_model, f)

print('count train target:')
train_val = np.zeros(df.ID.values.shape[0])

def get_train_res(tr):
	index_lst = []
	target_lst = []
	col1 = [tr[n] for n in range(args.nlet)]
	col2 = [tr[n+args.nlet] for n in range(args.nlet)]
	col3 = [tr[args.nlet*2]]
	for a in df[col1+['ID']+col3].values:
		if not 0 in a[0:args.nlet]:
			for b in df[col2+['ID']].values:
				if (b[0:args.nlet]==a[0:args.nlet]).all() and a[args.nlet]!=b[args.nlet]:
					index_lst.append(df[df['ID']==b[args.nlet]].index[0])
					target_lst.append(a[args.nlet+1])
	return index_lst, target_lst

proc_pool = Pool(args.num_process)

result = proc_pool.map(get_train_res, nlet_model)
for i, w in result:
	train_val[i] = w
	
print('%d/%d columns filled in train:'%(train_val[train_val!=0].shape[0],train_val.shape[0],))
