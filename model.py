import pandas as pd
import keras 
from train.py import data_gen, parse_data_1D
import random
import os
import time
import glob
import sys
import keras

class PatientModel():

    def __init__(id_no, cv = 0.8):
        self.id = int(id_no)
        self.cv = cv
        self.safe_list = []
        self.train_list = []
        self.cv_list = []
        self.train_dir = "../train_"+str(self.id)+"/"
        self.model = keras.Sequential()
        
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
        sgd = keras.SGD(lr=0.0001, decay=1e-6, momentum=0.8, nesterov=False)
        self.model.compile(loss='binary_crossentropy',optimizer= sgd)
        
        safe = pd.read_csv('train_and_test_data_labels_safe.csv')
        safeset = set(safe[safe['safe'] == 1]['image'].values)
        
        for filename in glob.glob(self.train_dir+"*.mat"):
            if os.path.getsize(filename)> 55000 and filename.split('/')[1] in safeset:
                self.safe_list.append(filename)
        random.shuffle(self.safe_list)
        self.train_list = safe_list[:int(len(safe_list_1)*self.cv)]
        self.cv_list = safe_list[int(len(safe_list_1)*self.cv)+1 :]

    def train(self, nb_epochs=10,batch_size = 937000):
        self.model.fit_on_generator(data_gen(self.train_list,10),batch_size,nb_epochs))
    
    def write_predict(self):
        f = open(sub_file, 'w')
        f.write('File,Class\n')
        outfiles=glob.glob(self.train_dir+'*.mat')
        for files in outfiles:
            file_name=files.split('/')[1]
            X=np.array(parse_data_1D(files,256))
            prediction=self.model.predict_proba(X, verbose = 0).mean()
            f.write(file_name + ','+str(prediction)+'\n')
        f.close()
