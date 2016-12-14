import keras 

class PatientModel():

    def __init__(id_no, cv = 0.2):
        self.id = int(id_no)
        self.cv = cv
        self.train_dir = "../train_"+str(self.id)+"/"
        self.model = keras.Sequential()
        self.model.add(Flatten(input_shape=(1,16,256))
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
        
     def train(nb_epochs=10,batch_size = 937000):
        self.model.fit_on_generator(data_gen(train_list,10),batch_size,nb_epochs))
        
