#!/usr/bin/python3
import argparse

from keras.models import Sequential
from keras import optimizers
import numpy
import io
import random
import datetime
# import sqlite3
import nnModel
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
import keras_helpers
import math
import sys
from flask import Flask
from flask import send_file, make_response, send_from_directory,request
from flask import jsonify
import datetime

import tensorflow as tf
import sqlite3


from sklearn.metrics import classification_report, confusion_matrix
numpy.set_printoptions(suppress=True,linewidth=numpy.nan,threshold=numpy.nan)


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True,
        help="type of processing", choices=['train', 'test', 'predict', 'evaluate', 'predict-service'])
ap.add_argument("-id", "--modelid", required=True,
            help="modelid")
FLAGS, unparsed = ap.parse_known_args()



modelId = FLAGS.modelid
MAX_WORD_LENGTH = 18
MIN_WORD_LENGTH = 4
MIN_PREDICTION_LENGTH=3
LETTER_COUNT = 26
X_SHAPE = (MAX_WORD_LENGTH - 1, LETTER_COUNT)
Y_SHAPE = (LETTER_COUNT)
def loadWords():
    words = []
    with open("google-10000-english-usa.txt") as f:
        for (i,line) in enumerate(f):
            if len(line)>=4 and len(line)<=MAX_WORD_LENGTH:
                words.append(line)
    return words

# def letterToFeature(letter):
#     feature = numpy.zeros(LETTER_COUNT);
#     feature[ord(letter)] = 1
#     return feature;

def letterToFeatureIndex(letter):
    return ord(letter) - 97#ord('a')

def wordToData(letters):
    dataX = numpy.zeros(X_SHAPE)
    for i, letter in enumerate(letters):
        dataX[i, letterToFeatureIndex(letter)] = 1
    return dataX;

def generateTrainingItem(letters, prediction):
    dataX = wordToData(letters)
    dataY = numpy.zeros(Y_SHAPE)
    
    dataY[letterToFeatureIndex(prediction)] = 1
    return dataX, dataY


def genData2(words):
    count = len(words)
    x = [];
    y = [];
    for word in words:
        for i in range(MIN_PREDICTION_LENGTH, len(word)-1, 1):
            # print(word, word[:i], word[i])
            dataX, dataY = generateTrainingItem(word[:i], word[i])
            # print(dataX)
            # print(dataY)
            # print (word[:i])
            # print(word[i+1])
            x.append(dataX)
            y.append(dataY);

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    return x, y

def genData(size, f):
    m = size;
    x = numpy.zeros((m, 1));
    y = numpy.zeros((m, 1));

    for i in range(m):
        xi = random.uniform(-10.0,10.0);
        yi = f(xi)  
        x[i]=xi;
        y[i] = yi   
    return x, y


def loadData():
    words = loadWords();
    x_train, y_train = genData2(words)
    x_test, y_test = x_train[0:100,:,:], y_train[0:100,:]
    x_cv, y_cv = numpy.expand_dims(numpy.zeros(X_SHAPE), 0), numpy.expand_dims(numpy.zeros(Y_SHAPE), 0) 

    return x_train, y_train, x_test, y_test, x_cv, y_cv


def loadTestData():
    m = 1000;
    f = lambda x: math.sin(x)#x**2#10*x+5#1000*x*x#math.sin(x)
    x_train, y_train = genData(m, f)
    x_test, y_test = genData(int(m/5), f)
    x_cv, y_cv = genData(int(m/5), f)

    return x_train, y_train, x_test, y_test, x_cv, y_cv

def predict (word):

    x_data = numpy.expand_dims(numpy.zeros(X_SHAPE), 0);
    x_data[0] = wordToData(word);
    # print(x_data[0])
    y_pred = model.predict(x_data, batch_size=32)
    # print (y_pred)
    suggestions = []
    
    for i,p in enumerate(y_pred[0]):
        suggestions.append({'label':chr(i+97), 'prediction':p})

    suggestions.sort(key=lambda item:item['prediction'], reverse=True)
    return suggestions

if FLAGS.type=='train':
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-c", "--continue",
            help="continue from previous training")
    FLAGS, unparsed = ap.parse_known_args(unparsed)
    
    x_train, y_train, x_test, y_test, x_cv, y_cv = loadData()#loadAllData(),loadTestData
    # x_train = x_train[0:100,:,:,:]
    # y_train = y_train[0:100,:]
    # print (x_train.shape)
    # print (y_train.shape)
    # featureCount = x_train.shape[1]
    print (modelId)
    model = nnModel.createModel(modelId, X_SHAPE);
    print (model.summary())

    batch_size = 2000
    epochs = 300
    if  vars(FLAGS)['continue']:
        file = open("model-{}-last.epoch".format(modelId), 'r')
        initEpochs = int(file.read())
        epochs+=initEpochs
        file.close()
    else:
        initEpochs = 0
    # optimizer = keras_helpers.MyAdamOptimizer(initEpochs, lr=0.001);
    # optimizer = optimizers.MyRMSpropOptimizer(initEpochs, lr=3e-6);
    # optimizer = optimizers.SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = optimizers.SGD(lr=0.0003, decay=1e-6)
    optimizer = keras_helpers.MySGDOptimizer(initEpochs, lr=1e-0001,decay=1e-06, momentum=0.9, nesterov=True)#
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[keras_helpers.top_4_accuracy])#'categorical_accuracy'
    if  vars(FLAGS)['continue']:
        model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    
    # callback_test = MyCallback(x_cv, y_cv);
    # callback_train = MyCallback(x_train, y_train);
    tensorboard = keras_helpers.MultiClassTensorBoard(trainSet={'x':x_train, 'y':y_train}, testSet={'x':x_test, 'y':y_test},
                                        log_dir='/ml/tlogs',write_images=True, write_grads=True, histogram_freq=50)
    dataGraphTensorBoard = keras_helpers.DataGraphTensorBoard(
                                        log_dir='/ml/tlogs/training', trainSet={'x':x_train, 'y':y_train}, byIndex = False, frequency=500)
    
    dataValidationGraphTensorBoard = keras_helpers.DataGraphTensorBoard(
                                        log_dir='/ml/tlogs/validation', trainSet={'x':x_cv, 'y':y_cv}, byIndex = False, frequency=500)
    

    history = model.fit(x_train, y_train,  batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_cv, y_cv)#, 
                        ,callbacks=[tensorboard], initial_epoch = initEpochs)# callbacks=[callback_train, callback_test])#validation_data=(x_test, y_test),dataGraphTensorBoard,tensorboard
     
    # print (model.evaluate(x_cv, y_cv, verbose=1))
    file = open("model-{}-last.epoch".format(modelId), 'w')
    file.write("{}".format((epochs)));
    file.close()
    model.save_weights("model-{}-last.hdf5".format(modelId))
    # model.save_weights("model-{}-{}.hdf5".format(modelId,datetime.datetime.now()))

elif FLAGS.type=='evaluate':
    
    model = nnModel.createModel(modelId, X_SHAPE);
    x_test_pos, y_test_pos = loadData(2, 1, 1500);
    x_test_neg, y_test_neg = loadData(2, 0, 2000);
    x_test_neg2, y_test_neg2 = loadSingleLineAsNegativeData(2, 1000);
    # x_test_q, y_test_q =     loadData(2,  100, 1);

    x_test = numpy.concatenate((x_test_pos, x_test_neg, x_test_neg2))
    y_test = numpy.concatenate((y_test_pos, y_test_neg, y_test_neg2))

    batch_size = 32
    model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    optimizer = optimizers.Adam(lr=0.000003);

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    print(model.evaluate(x_test, y_test))
    predictions = model.predict(x_test, batch_size=32)
    print (calculateScore(predictions, y_test))

elif FLAGS.type=='predict':

    # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
    model = nnModel.createModel(modelId, X_SHAPE);
    model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    while(True):
        line = sys.stdin.readline()
        word = line[:len(line)-1]
        if len(word)<MIN_PREDICTION_LENGTH or len(word)>MAX_WORD_LENGTH:
            print("word length should be between {}-{}, actual value is {}".format(MIN_PREDICTION_LENGTH, MAX_WORD_LENGTH, len(word)))
            continue
       
        suggestions = predict(word)
        
        for suggestion in suggestions[:4]:
            print ("{} : {:.2f}".format(suggestion['label'], suggestion['prediction'])) 
    
elif FLAGS.type=='predict-service':

    # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
    model = nnModel.createModel(modelId, X_SHAPE);
    model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);

    app = Flask(__name__)
    graph = tf.get_default_graph()

    @app.route("/")
    def root():
        return "Welcome!\nusage: /prediction/min_3_letter"

    @app.route("/prediction/<string:str>", methods=['GET'])
    def prediction(str):

        if(MIN_PREDICTION_LENGTH<=len(str)<=MAX_WORD_LENGTH):
            global graph
            with graph.as_default():
                suggestions = predict(str)
                response = '';
                for suggestion in suggestions[:4]:
                    response += "{} : {:.2f}<br/>".format(suggestion['label'], suggestion['prediction'])
                return response
        else:
            return "Bad Request, word length should be between {}-{}, actual value is {}".format(MIN_PREDICTION_LENGTH, MAX_WORD_LENGTH, len(str)), 400
    

    @app.route("/pattern/<string:reportid>/<string:rowCategories>/<string:columnCategories>", methods=['GET'])
    def addPattern(reportid, rowCategories, columnCategories):
        now = datetime.datetime.now()
        with sqlite3.connect('/ml/patterns.sqlite') as conn:
            c = conn.cursor()
            c.execute('insert into patterns (reportid, rowPattern, columnPattern, datetime) values(?,?,?,?)', [reportid, rowCategories, columnCategories, now.strftime("%Y-%m-%d %H:%M")])
            conn.commit()
        return "OK"

    @app.route("/patterns", methods=['GET'])
    def getPatterns():
        with sqlite3.connect('/ml/patterns.sqlite') as conn:
            c = conn.cursor()
            patterns = c.execute('SELECT * FROM patterns').fetchall()
        return jsonify(patterns)

    app.run(debug=True, host= '0.0.0.0')

elif FLAGS.type=='test':
    # words = loadWords()
    maxlen = 0;
    # for word in words:
    #     if len(word)>maxlen:
    #         maxlen=len(word)
    # print(maxlen)
    # x_train, y_train, x_test, y_test, x_cv, y_cv = loadAllData()
    # data = {};
    # for x, y in zip(x_train, y_train):
    #     key = "_".join(map(str,x));
    #     value = y
    #     if key in data:
    #         l = data[key]
    #     else:
    #         l = [];
    #         data[key] = l
    #     l.append(y)
    # print (len(x_train))
    # for k, v in data.items():
    #     if len(v)>1:
    #         print (v)
    # model = nnModel.createModel(modelId);
    # model.load_weights("model-{}-last.hdf5".format(modelId), by_name=False);
    # predictions = model.predict(x_train, batch_size=32)
    # m = len(x_train)
    # z = numpy.zeros((m, 2));
    
    # z[:,0] = y_train[:, 0]
    # z[:,1] = predictions[:,0]
    # result = numpy.append(x_train, z, 1);

    # numpy.savetxt('result.csv',result, delimiter=',',fmt='%10.5f')

    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print (weights[0:10])



