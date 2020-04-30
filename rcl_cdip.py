import os
from pathlib import Path
import numpy as np

from tensorflow.keras.utils import to_categorical

import kegra
import kegra.utils

import scipy.sparse as sp


def load_one(identifier = "00043445_00043449", data_dir="grapher_outputs/numpy/"):
    X = np.load(os.path.join(data_dir, identifier+"_X.npy"))
    X = X.reshape(-1, *X.shape)
    A = np.load(os.path.join(data_dir, identifier+"_A.npy"))
    A = A.reshape(-1, *A.shape)
    Y = np.load(os.path.join(data_dir, identifier+"_Y.npy"))
    Y = Y.reshape(-1, *Y.shape)
    return X, A, Y


def get_splits(y,
               idx_train = range(140),
               idx_val = range(200, 500),
               idx_test = range(500, 1500)):
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = kegra.utils.sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def load(data_dir="grapher_outputs/numpy/", max_doc=1000):
    n = 0
    print("Loading data...")
    for n, f in enumerate(Path(data_dir).glob('*_A.npy')):
        if n >= max_doc:
            break
        file_identifier = f.stem[:-2]
        Xf, Af, Yf = load_one(file_identifier, data_dir)
        if n == 0:
            X, A, Y = Xf, Af, Yf
        else:
            X = np.concatenate([X, Xf])
            A = np.concatenate([A, Af])
            Y = np.concatenate([Y, Yf])
    print(f"Loaded {n+1} documents")
    return X, A, to_categorical(Y)


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = np.concatenate([np.eye(adj.shape[1]).reshape(-1, *a.shape) for a in adj])
    adj = normalize_adj(adj, symmetric)
    return adj

