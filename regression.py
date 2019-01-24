import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import History 
import matplotlib.pyplot as plt
from keras.layers import Input, Concatenate, ZeroPadding1D
from keras.layers import Activation, Conv1D, GlobalAveragePooling1D, AveragePooling1D, MaxPooling1D,BatchNormalization, add, UpSampling1D


def plotResults(model, X_train, X_test, y_train, y_test, history, pointsToShow):
    predicted = model.predict(X_test)
    training = model.predict(X_train)
    predicted_score = model.evaluate(X_test, y_test)
    training_score  = model.evaluate(X_train, y_train)
    plt.rcParams.update({'font.size': 14})
    fig= plt.figure()
    ax0= fig.add_subplot(3,1,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    ax0.set_title("Training Loss vs Validation Loss")
    ax0.legend(loc="best")
    ax1= fig.add_subplot(3,1,2)
    index = np.random.choice(np.arange(training.shape[0]),pointsToShow-1)
    print(index.shape)
    plt.plot(np.arange(1,training.shape[0]+1,1)[1:pointsToShow], training[index].flatten(), label="model")
    plt.plot(np.arange(1,y_train.shape[0]+1,1)[1:pointsToShow], y_train[index].flatten(), label="real")
    ax1.set_title("Fit on training data " + "MSE=" + str(round(training_score,2)))
    ax1.legend(loc="best")
    ax2= fig.add_subplot(3,1,3)
    index2 = np.random.choice(np.arange(predicted.shape[0]),pointsToShow-1)
    plt.plot(np.arange(1,predicted.shape[0]+1,1)[1:pointsToShow], predicted[index2].flatten(), label="model")
    plt.plot(np.arange(1,y_test.shape[0]+1,1)[1:pointsToShow], y_test[index2].flatten(), label="real")
    ax2.set_title("Fit on holdout data " + "MSE=" + str(round(predicted_score,2)))
    ax2.legend(loc="best")
    plt.tight_layout()   

#described here: https://arxiv.org/pdf/1611.06455.pdf
def conv_model(feature_size, act):
    inputs = Input(shape=(feature_size,1))
    conv1 =  Conv1D(128, 8, padding='same')(inputs)
    bn1   =  BatchNormalization()(conv1)
    act1  =  Activation(act)(bn1)
    conv2 =  Conv1D(256, 5, padding='same')(act1)
    bn2   =  BatchNormalization()(conv2)
    act2  =  Activation(act)(bn2)
    conv3 =  Conv1D(128, 3, padding='same' )(act2)
    bn3   =  BatchNormalization()(conv3)
    act3  =  Activation(act)(bn3)
    gap   = GlobalAveragePooling1D()(act3)
    dense = Dense(1, activation=act)(gap)
    opt =  Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss='mse', optimizer=opt)
    return model

#described here: https://arxiv.org/pdf/1611.06455.pdf
def resnet_model(feature_size, act):
    inputs = Input(shape=(feature_size,1))
    conv1 =  Conv1D(64, 8, padding='same')(inputs)
    bn1   =  BatchNormalization()(conv1)
    act1  =  Activation(act)(bn1)
    # second conv layer
    conv2 = Conv1D(64, 5, padding='same')(act1)
    bn2   = BatchNormalization()(conv2)
    act2  = Activation(act)(bn2)
    # third conv layer
    conv3 = Conv1D(64, 3, padding="same")(act2)
    bn3   = BatchNormalization()(conv3)
    z1    = add([inputs, bn3])
    act3  = Activation(act)(z1)    
    # second ResNet block
    conv4 =  Conv1D(128, 8, padding='same')(act3)
    bn4   =  BatchNormalization()(conv4)
    act4  =  Activation(act)(bn4)
    # second conv layer
    conv5 = Conv1D(128, 5, padding='same')(act4)
    bn5   = BatchNormalization()(conv5)
    act5  = Activation(act)(bn5)
    # third conv layer
    conv6 = Conv1D(128, 3, padding='same')(act5)
    bn6   = BatchNormalization()(conv6)
    act31 = Conv1D(128, 1, padding='same')(act3)  #matching dimensions
    z2    = add([act31, bn6])
    act6  = Activation(act)(z2)    
    # third ResNet block
    conv7 =  Conv1D(128, 8, padding='same')(act6)
    bn7   =  BatchNormalization()(conv7)
    act7  =  Activation(act)(bn7)
    # second conv layer
    conv8 = Conv1D(128, 5, padding='same')(act7)
    bn8   = BatchNormalization()(conv8)
    act8  = Activation(act)(bn8)
    # third conv layer
    conv9 = Conv1D(128, 3, padding='same' )(act8)
    bn9   = BatchNormalization()(conv9)
    z3    = add([act6, bn9])
    act9  = Activation(act)(z3)    
    # global pooling & softmax
    gap   = GlobalAveragePooling1D()(act9)
    dense = Dense(1, activation=act)(gap)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)  #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss='mse', optimizer=opt)
    return model

def train(dataset, currents):
    feature_size = currents * dataset
    dataset = np.loadtxt("\\data\\sample7000x" +str(dataset)+".csv", delimiter=",")
    X = dataset[:,0:feature_size]
    Y = dataset[:,dataset.shape[1]-1]
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    "use only 1.5, 5.5 and 10.5 for training and the rest is used for testing"
    X_train = X[(Y==1.5) | (Y==5.5) | (Y == 10.5)]
    X_test  = X[(Y==2.5) | (Y==3.5) | (Y == 4.5) | (Y == 6.5)| (Y == 7.5) | (Y == 8.5) | (Y == 9.5)]
    y_train = Y[(Y==1.5) | (Y==5.5) | (Y == 10.5)]
    y_test  = Y[(Y==2.5) | (Y==3.5) | (Y == 4.5) | (Y == 6.5)| (Y == 7.5) | (Y == 8.5) | (Y == 9.5)]
    X_train =  np.expand_dims(X_train, axis=2)
    X_test  =  np.expand_dims(X_test, axis=2)
    model = conv_model(feature_size, "relu")
    history = History()
    model.fit(X_train, y_train, epochs=50,  batch_size=120, validation_data=(X_test, y_test), callbacks=[history])
    plotResults(model, X_train, X_test, y_train, y_test, history, 100)


datasets=[4,6,8,32,64]
currents=[1,2,3]

for i in datasets:
    for j in currents:
        print("Dataset="+str(i))
        print("Currents="+str(j))
        train(i,j)
