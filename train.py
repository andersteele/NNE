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

def data_gen(path_list, n = 1):
    #Parses files in list (n at a time), mixing he samples
    M=len(path_list)-n-1
    while True:
        i=random.randint(0,M)
        train_out=[]
        x_out=parse_data_1D(path_list[i])
        y_out=np.ones(x_out.shape[0])*int((path_list[i].split('/')[1]).split('.')[0].split('_')[2])
        for paths in path_list[i+1:i+n]:
            temp_x=parse_data_1D(paths)
            x_out=np.vstack([x_out,temp_x])
            y_out=np.hstack([y_out,[int((paths.split('/')[1]).split('.')[0].split('_')[2])]*len(temp_x)])
        p = np.random.permutation(x_out.shape[0])
        yield(np.take(x_out,p,axis=0),y_out[p])
                 
