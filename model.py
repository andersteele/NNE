import pandas as pd
import numpy as np
import keras 
import random
import os
import time
import glob
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from scipy.io import loadmat

def parse_data_1D(path, T = 256):
    try:
        raw = loadmat(path)['dataStruct']['data'][0,0]
    except:
        print('bad file: '+path)
    delta = T
    M = int((raw.shape[0]/T))
    C = raw.shape[1]
    out = np.empty((M,1,C,T))
    for i in range(M):
        out[i,0,:,:] = raw[delta*i:delta*(i+1),:].transpose()
    return out

def data_gen(path_list, n = 1):
    #Parses files in list (n at a time), mixing he samples
    M = len(path_list)-n-1
    while True:
        i=random.randint(0,M)
        train_out=[]
        x_out=parse_data_1D(path_list[i])
        y_out=np.ones(x_out.shape[0])*int((path_list[i].split('/')[2]).split('.')[0].split('_')[2])
        for paths in path_list[i+1:i+n]:
            temp_x=parse_data_1D(paths)
            x_out=np.vstack([x_out,temp_x])
            y_out=np.hstack([y_out,[int((paths.split('/')[2]).split('.')[0].split('_')[2])]*len(temp_x)])
        p = np.random.permutation(x_out.shape[0])
        yield(np.take(x_out,p,axis=0),y_out[p])


class PatientModel():

    def __init__(self, id_no, cv = 0.8):
        self.cv = cv
        self.safe_list = []
        self.train_list = []
        self.cv_list = []
        self.train_dir = "../train_"+str(id_no)+"/"
        self.model = Sequential()
        
        self.model.add(Flatten(input_shape=(1,16,256))) #a simple NN with 3 hidden layers
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.8, nesterov=False)
        self.model.compile(loss='binary_crossentropy',optimizer= sgd)
        
        safe = pd.read_csv('../train_and_test_data_labels_safe.csv')
        safeset = set(safe[safe['safe'] == 1]['image'].values)
        
        for filename in glob.glob(self.train_dir+"*.mat"):
            if os.path.getsize(filename)> 55000 and filename.split('/')[2] in safeset:
                self.safe_list.append(filename)
        random.shuffle(self.safe_list)
        self.train_list = self.safe_list[0:int(len(self.safe_list)*self.cv)]
        self.cv_list = self.safe_list[int(len(self.safe_list)*self.cv)+1 :]

    def train(self, nb_epochs=10,batch_size = 937000):
        self.model.fit_generator(data_gen(self.train_list,10),batch_size,nb_epochs)
    
    def write_predict(self):
        f = open(sub_file, 'w')
        f.write('File,Class\n')
        outfiles=glob.glob(self.train_dir+'*.mat')
        for files in outfiles:
            file_name=files.split('/')[2]
            X=np.array(parse_data_1D(files,256))
            prediction=self.model.predict_proba(X, verbose = 0).mean()
            f.write(file_name + ','+str(prediction)+'\n')
        f.close()
