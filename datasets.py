import re
import os

import numpy as np
import lightgbm as lgb

from pathlib import Path
from itertools import groupby
from collections import Counter


def init_path(data_path='../../datasets/numpy_datasets/', result_path = '../', project_name = 'ALL'):
    output_path = os.path.join(result_path, 'output/')
    models_path = os.path.join(output_path, 'models_' + project_name + '/')
    results_path = os.path.join(output_path, 'results_'+ project_name + '/')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    return output_path, models_path, results_path, data_path

def init_dataset(data_path, param_text):
    print('[INFO] Loading data:')
    train_data, train_labels, train_query_lens = load_data_npy(data_path + param_text + '_train.npy')
    print('[INFO] Training set loaded:')
    valid_data, valid_labels, valid_query_lens = load_data_npy(data_path + param_text + '_valid.npy')
    print('[INFO] Validation set loaded:')
    test_data, test_labels, test_query_lens = load_data_npy(data_path + param_text + '_test.npy')
    print('[INFO] Testing set loaded:')

    return train_data, train_labels, train_query_lens, valid_data, valid_labels, valid_query_lens, test_data, test_labels, test_query_lens

def load_data_npy(filename):
    X = np.load(filename)

    data = X[:,:-2]
    labels = X[:,-2]
    query_lens = np.asarray([len(list(group)) for key, group in groupby(X[:,-1])])

    return data, labels.astype(int), query_lens.astype(int)

def remove_docs(data, labels, query_lens, idx_to_remove):
    idx_to_keep = np.setdiff1d(np.arange(data.shape[0]), idx_to_remove)
    qs_lens = np.repeat(np.arange(len(query_lens)), query_lens)
    new_qs_lens = np.bincount(qs_lens[idx_to_keep])

    return data[idx_to_keep], labels[idx_to_keep], new_qs_lens

# split dataset
def split_dataset_by_query(X, y, g, s):
    size = np.sum(g[:-s])
    return X[:size], y[:size], g[:-s], X[size:], y[size:], g[-s:]

# return a specific query
def get_query_by_id(X, y, q, idx):
    c_sum = np.insert(np.cumsum(q), 0, 0)
    start = c_sum[idx]
    end = start + q[idx]

    return np.array(X[start: end, :]), y[start: end], [q[idx]]

# normalise dataset by queries
def normalise_queries(data, qs_len):
    qs_range = np.cumsum(qs_len)
    qs_range = np.concatenate([np.zeros(1, dtype=qs_range.dtype), qs_range])
    
    for i in range(qs_len.shape[0]):
        s_i, e_i = qs_range[i : i + 2]
        cur_feat = data[s_i : e_i, :]
        min_q = np.amin(cur_feat, axis=0)
        max_q = np.amax(cur_feat, axis=0)
        cur_feat -= min_q[None, :]
        denom = max_q - min_q
        denom[denom == 0.] = 1.
        cur_feat /= denom[None, :]
        data[s_i : e_i, :] = cur_feat

    return data

def dataset_statistics(data, labels, qlen):
    n_ists, n_feat = data.shape
    n_query = len(qlen)
    print('#ist: ', n_ists, ' #feat: ', n_feat, '#query: ', n_query)
    
    max_grade = np.max(labels)
    sort = [0] * (max_grade + 1)
    occ = Counter(labels)
    for x,y in occ.items():
        sort[x] = (x, y, y / n_ists * 100)
        
    for data in sort:
        print(data)

def save_in_numpy(data, labels, query_lens, filename):
    X = data
    y = labels.reshape(-1, 1)

    q = []
    for i, q_len in enumerate(query_lens):
        q += [i] * q_len
    q = np.asarray(q).reshape(-1, 1)

    data = np.hstack((X,y,q))
    np.save(filename, data)

def save_svmlight_to_numpy(data, labels, query_lens, filename, zero_cols=False, f_index=None):
    data = data.toarray()
    y = labels.reshape(-1, 1)
    q = query_lens.reshape(-1, 1)
    
    print('Dataset shape')
    print('- input shape:', data.shape, labels.shape, query_lens.shape)
    
    if f_index is not None:
        print('Select features')
        print('- original shape', data.shape)
        data = data[:, f_index]
        print('- output shape', data.shape)
        
    if zero_cols:
        print('Remove zero columns')
        print('- original shape', data.shape)
        m = data.any(axis=0)
        data = data[:, m]
        print('- output shape', data.shape)

    print('- output shape:', data.shape, labels.shape, query_lens.shape)
    dataset = np.hstack((data, y, q))
    np.save(filename, dataset)

# reads svmlight file and retrives right feature indices
def get_feature_index(filename, index=None):
    pattern = ' ([0-9]*):'

    if index is None:
        index = np.empty((0, ), int)

    with open(filename, 'r', encoding="ISO-8859-1") as f:
        for line in f:
            out = re.findall(pattern, line)
            out = np.unique(np.asarray(out).astype(int))
            index = np.union1d(index, out)
    
    return index

def prepare_lightgbm_sets(train_set, eval_set=None, include_train=False):
    train_set = lgb.Dataset(train_set[0], train_set[1], group=train_set[2], params={"name" : "train"})

    valid_sets = []
    valid_names = []
    if include_train:
        valid_sets = [train_set]
        valid_names = ["train"]

    if eval_set is not None:
        valid_sets += [lgb.Dataset(ds[0], ds[1], group=ds[2], reference=train_set, params={'name' : ds[3]}) for ds in eval_set]
        valid_names += [ds[3] for ds in eval_set]

    return train_set, valid_sets, valid_names
