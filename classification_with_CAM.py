import keras
import random
import numpy as np
import pandas as pd
from time import time
import matplotlib as mpl
import matplotlib.cm as cm
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Model
from keras.callbacks import History 
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.collections import LineCollection
from keras.layers import Dense, Activation, Conv1D, GlobalAveragePooling1D, BatchNormalization, Input, add, UpSampling1D

def plotResults(model, X_train, X_test, y_train, y_test, history, pointsToShow):
    predicted = model.predict(X_test)
    training = model.predict(X_train)
    predicted_score = model.evaluate(X_test, y_test)
    training_score  = model.evaluate(X_train, y_train)
    fig= plt.figure()
    #history
    ax0= fig.add_subplot(3,1,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    ax0.set_title("Training Loss vs Validation Loss")
    ax0.legend(loc="best")
    '''ax1= fig.add_subplot(3,1,2)
    plt.plot(np.arange(1,training.shape[0]+1,1)[1:pointsToShow], training[1:pointsToShow], label="model")
    plt.plot(np.arange(1,y_train.shape[0]+1,1)[1:pointsToShow], y_train[1:pointsToShow], label="real")
    ax1.set_title("Fit on training data " + "Accuracy=" + str(training_score))
    ax1.legend(loc="best")
    ax2= fig.add_subplot(3,1,3)
    plt.plot(np.arange(1,predicted.shape[0]+1,1)[1:pointsToShow], predicted[1:pointsToShow], label="model")
    plt.plot(np.arange(1,y_test.shape[0]+1,1)[1:pointsToShow], y_test[1:pointsToShow], label="real")
    ax2.set_title("Fit on holdout data " + "Accuracy=" + str(predicted_score))
    ax2.legend(loc="best")'''
    plt.tight_layout()   


#global
batch_size = 120
epochs = 50  #1500

# data set
dataset=128
currents=2

history = History()

#location
dir_path=""+ str(dataset)+"\\"

train = pd.read_table(dir_path+"generator_TRAIN.csv", sep = ",", header = 0)
test  = pd.read_table(dir_path+"generator_TEST.csv", sep = ",", header = 0)

#feature_size= train.shape[1]-1
feature_size = currents * dataset
print(feature_size)

train_X =  train.iloc[:,0:feature_size].values
test_X  =  test.iloc[:,0:feature_size].values
    
encoder = LabelEncoder()
encoder.fit(train.iloc[:,train.shape[1]-1])
encoded_Y1 = encoder.transform(train.iloc[:,train.shape[1]-1])
encoded_Y2 = encoder.transform(test.iloc[:,test.shape[1]-1])

train_Y = np_utils.to_categorical(encoded_Y1)
test_Y =  np_utils.to_categorical(encoded_Y2)
        
train_X =  np.expand_dims(train_X, axis=2)
test_X  =  np.expand_dims(test_X, axis=2)
nr_classes= train_Y.shape[1]

inputs = Input(shape=(feature_size,1))

# first conv layer
conv1 =  Conv1D(128, 8, padding='same')(inputs)
bn1   =  BatchNormalization()(conv1)
act1  =  Activation('relu')(bn1)
# second conv layer
conv2 =  Conv1D(256, 5, padding='same')(act1)
bn2   =  BatchNormalization()(conv2)
act2  =  Activation('relu')(bn2)
# third conv layer
conv3 =  Conv1D(128, 3, padding='same' )(act2)
bn3   =  BatchNormalization()(conv3)
act3  =  Activation('relu')(bn3)
# global pooling & softmax

gap   = GlobalAveragePooling1D()(act3)
dense = Dense(nr_classes, activation='softmax')(gap)
opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model = Model(inputs=inputs, outputs=dense)
model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_Y,  epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_Y),callbacks=[history])


# Score trained model.
scores = model.evaluate(test_X, test_Y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

y_pred = model.predict(test_X); y_pred = np.argmax(y_pred, axis=1)
c=confusion_matrix(encoder.inverse_transform(encoded_Y2), encoder.inverse_transform(y_pred))
print(classification_report(encoder.inverse_transform(encoded_Y2), encoder.inverse_transform(y_pred)))

   
# Class Activation Map - CAM - see what the network is seeing
random.seed(10)
for i in range(9):
    fig, ax = plt.subplots()
    ax.set_title("Class= "+str(i))
    cmap = plt.get_cmap('rainbow')
    k=1
    while k<=1:
        current = random.randint(1,test_X.shape[0])-1
        if np.argmax(test_Y[current])==i:
            k=k+1 
            example = np.expand_dims(test_X[current,:,:], axis=0);
            activation = K.function([model.layers[0].input],[model.layers[-3].output, model.layers[-1].output])
            [activation_output, prediction] = activation([example])
            weightIndex= np.argmax(prediction)
            weights = model.layers[-1].get_weights()[0]
            t = np.dot(activation_output[0,:,:], weights[:,weightIndex]) 
            t = t.reshape(feature_size)
            x=np.arange(example.shape[1])
            y=example.flatten()
            ax.set_xlim(x.min()-1, x.max()+1)
            ax.set_ylim(y.min()-1, y.max()+2)
            points = np.array([x, y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = mpl.colors.Normalize(vmin=min(t), vmax=max(t))
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(t)
            ax.add_collection(lc)

    plt.show()
    
