## Routines for parsing training data and training of neural network
##
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import random
import os
import time
import glob
import sys
import keras

def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata
def parse_data(path):
    raw=mat_to_data(path)['data']
    out = []
    delta = 256
    M = int((raw.shape[0]/256.0))
    for i in range(M):
        out.append(np.array([raw[delta*i:delta*(i+1),:]]))
    return np.array(out)
def parse_data_1D(path, T = 256):
    try:
        raw=loadmat(path)['dataStruct']['data'][0,0]
    except:
        print('bad file: '+path)
    delta = T
    M = int((raw.shape[0]/T))
    C=raw.shape[1]
    out=np.empty((M,1,C,T))
    for i in range(M):
        out[i,0,:,:]=raw[delta*i:delta*(i+1),:].transpose()
    return out
def create_train_list(dirs = 'train_1/', no_files=100, seg_size=256):
    #dir specifies patient
    #batch_size is number of files to process at a time
    #length is the size of
    train_list=[]
    file_list=glob.glob(dirs+'*.mat')
    raw_0 = loadmat(file_list[0])['dataStruct']['data'][0,0].transpose()
    M=int(raw_0.shape[1]/seg_size)
    x = np.empty((1,16,seg_size))
    for paths in [file_list[i] for i in range(no_files)]:
        #add safety checking
        raw = loadmat(paths)['dataStruct']['data'][0,0].transpose()
        y = int((paths.split('/')[1]).split('.')[0].split('_')[2])
        for j in range(M):
            x[0,:,:]=raw[:,j*seg_size:(j+1)*seg_size]
            train_list.append((x,y))
    random.shuffle(train_list)
    return train_list
