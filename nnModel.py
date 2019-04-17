#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D,LSTM,Activation
from keras import optimizers


# (FEATURE_COUNT) = (802)#(802)

def createModel1(inputShape):
    model = Sequential()
    model.add(Dense(5, activation='sigmoid', kernel_regularizer=None, kernel_initializer='uniform', input_shape=inputShape))#kernel_regularizer=None, 
    model.add(Dense(15, activation='sigmoid', kernel_regularizer=None, kernel_initializer='uniform'))
    model.add(Dense(5, activation='sigmoid', kernel_regularizer=None, kernel_initializer='uniform'))
    # model.add(Dense(10, activation='sigmoid', kernel_regularizer=None, kernel_initializer='uniform'))
    # model.add(Dense(10, activation='sigmoid', kernel_regularizer=None, kernel_initializer='uniform'))
    # model.add(Dense(2, activation='relu', kernel_regularizer=None, kernel_initializer='normal'))
    # model.add(Dense(2, activation='relu', kernel_regularizer=None, kernel_initializer='normal'))
    # model.add(Dense(2, activation='relu', kernel_regularizer=None, kernel_initializer='normal'))
    # model.add(Dense(2, activation='relu', kernel_regularizer=None, kernel_initializer='normal'))
    # model.add(Dense(2, activation='relu', kernel_regularizer=None, kernel_initializer='normal'))
    # model.add(Dense(200, activation='sigmoid', kernel_initializer='normal'))
    # model.add(Dense(200, activation='sigmoid', kernel_initializer='normal'))
    # model.add(Dense(1200, activation='relu'))
    # model.add(Dense(1200, activation='relu'))
    # model.add(Flatten())

    model.add(Dense(1, activation='linear'))
    return model

def createModel2(inputShape):
    model = Sequential()
    model.add(Dense(6400, activation='relu', input_shape=inputShape))
    model.add(Dense(3200, activation='relu'))
    model.add(Dense(3200, activation='relu'))
    model.add(Dense(3200, activation='relu'))
    # model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    return model

def createModel3(inputShape):
    model = Sequential()
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='uniform', input_shape=inputShape))
    model.add(Dense(2000, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dense(3000, activation='tanh', kernel_initializer='uniform'))
    model.add(Dense(2000, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dense(3000, activation='tanh', kernel_initializer='uniform'))
    model.add(Dense(3000, activation='sigmoid', kernel_initializer='uniform'))
    # model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    return model



def createModel4(inputShape):
    model = Sequential()
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='uniform', input_shape=inputShape))
    model.add(Dense(2000, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(3000, activation='tanh', kernel_initializer='uniform'))
    model.add(Dense(2000, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(3000, activation='tanh', kernel_initializer='uniform'))
    model.add(Dropout(0.10))
    model.add(Dense(3000, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dropout(0.10))
    # model.add(Flatten())
    model.add(Dense(1, activation='linear'))


    return model

def createModel5(inputShape):
    model = Sequential()
    model.add(Dense(50000, activation='sigmoid', input_shape=inputShape))
    model.add(Dense(5000, activation='sigmoid'))
    model.add(Dense(4000, activation='sigmoid'))
    model.add(Dense(4000, activation='sigmoid'))
    model.add(Dense(4000, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    return model

def createModel6(inputShape):
    model = Sequential()
    model.add(Dense(2000, activation='sigmoid', input_shape=inputShape))
    model.add(Dense(3000, activation='sigmoid'))
    model.add(Dense(4000, activation='sigmoid'))
    model.add(Dense(5000, activation='sigmoid'))
    model.add(Dense(6000, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    return model
#multi label classifer ~72% accuracy on autocomplete, 1600 epochs
#94% with top_3_accuracy
#88% with top_2_accuracy
def createModel7(inputShape):
    model = Sequential()
    model.add(LSTM(100, input_shape=(inputShape)))
    model.add(Dense(26, activation='sigmoid'))
    return model

#multi label classifer ~72% accuracy on autocomplete, 600 epochs
#94% with top_3_accuracy
#88% with top_2_accuracy
def createModel8(inputShape):
    model = Sequential()
    model.add(LSTM(300, input_shape=(inputShape)))
    model.add(Dense(26, activation='sigmoid'))
    return model




#multi label classifer ~70% accuracy on autocomplete, 300 epochs, fast
#96% with top_4_accuracy
#94% with top_3_accuracy
#88% with top_2_accuracy
def createModel9(inputShape):
    model = Sequential()
    model.add(Conv1D(100, 4, padding='valid', activation='relu', strides=1, input_shape=inputShape))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Flatten())
    model.add(Dense(26, activation='sigmoid'))
    return model
#multi label classifer ~70% accuracy on autocomplete
#93% with top_3_accuracy
#86% with top_2_accuracy
def createModel10(inputShape):
    model = Sequential()
    model.add(Conv1D(30, 3, padding='valid', activation='relu', strides=1, input_shape=inputShape))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Flatten())
    model.add(Dense(26, activation='sigmoid'))
    return model

def createModel(id, inputShape):
    f = eval('createModel{}'.format(id))
    return f(inputShape)
    # if id==1:
    #     return createModel1((HEIGHT,WIDTH, 1), 2);
    # elif id==2:
    #     return createModel2((HEIGHT,WIDTH, 1), 2);
    # elif id==3:
    #     return createModel3((HEIGHT,WIDTH, 1), 2);



