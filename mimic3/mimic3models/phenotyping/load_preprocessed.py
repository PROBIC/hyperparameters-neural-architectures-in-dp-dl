import os

import numpy as np


def load_cached_data(datapath):

    if datapath is None:
        datapath = os.path.join(os.path.dirname(__file__), "../../data/phenotyping/")

    #print("Load data from", datapath)
    train_X = np.load(os.path.join(datapath, "train_X.npy"))
    train_y = np.load(os.path.join(datapath, "train_y.npy"))
    val_X = np.load(os.path.join(datapath, "val_X.npy"))
    val_y = np.load(os.path.join(datapath, "val_y.npy"))
    test_X = np.load(os.path.join(datapath, "test_X.npy"))
    test_y = np.load(os.path.join(datapath, "test_y.npy"))

    #print("Data sucessfully loaded.")

    return train_X, train_y, val_X, val_y, test_X, test_y
