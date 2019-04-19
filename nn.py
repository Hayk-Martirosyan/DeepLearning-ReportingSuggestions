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

ROW_MAX_PATTERN_LENGTH = 8
COL_MAX_PATTERN_LENGTH = 10

ALL_OBJECTS = {
    "Action Activity":"290",
    "Action Partner":"289",
    "Action Status":"284",
    "Activity":"214",
    "Activity Actions":"293",
    "Activity Name":"219",
    "Activity Type":"218",
    "Actual Sub-Thematic Area":"216",
    "Actual Thematic Area":"215",
    "Agreement Created?":"297",
    "Agreement Currency":"298",
    "Agreement Status":"296",
    "Association":"800",
    "Association Cohort":"822",
    "Association Country":"808",
    "Association Households":"802",
    "Association Individuals":"801",
    "Association Region":"807",
    "Association Sub-Division 1":"809",
    "Association Sub-Division 2":"810",
    "Association Sub-Division 3":"811",
    "Budget Item":"275",
    "Budget Month":"280",
    "Budget SOF":"292",
    "Budget Year":"279",
    "Category 1":"259",
    "Category 2":"260",
    "Category 3":"261",
    "Code - Account Name":"276",
    "Cohort":"522",
    "Common Approach":"210",
    "Community":"700",
    "Community Country":"708",
    "Community Households":"702",
    "Community Individuals":"701",
    "Community Region":"707",
    "Community Sub-Division 1":"709",
    "Community Sub-Division 2":"710",
    "Community Sub-Division 3":"711",
    "Context":"209",
    "Contract Type":"299",
    "Cost Center":"278",
    "Cost Centers":"282",
    "Country":"203",
    "Cross-cutting Themes":"283",
    "Day":"325",
    "Disability":"503",
    "Donor Approval Required For":"273",
    "Draft Cross-cutting Themes":"212",
    "Employee":"285",
    "Employee Type":"271",
    "Estimated Sub-Thematic Area":"205",
    "Estimated Thematic Area":"204",
    "Gender":"502",
    "Household":"600",
    "Household Cohort":"622",
    "Household Country":"604",
    "Household Head":"601",
    "Household Individuals":"623",
    "Household Region":"603",
    "Household Sub-Division 1":"605",
    "Household Sub-Division 2":"606",
    "Household Sub-Division 3":"607",
    "Household Vulnerability":"602",
    "HR Plan ID":"269",
    "Humanitarian Response Category":"221",
    "Humanitarian Response Code":"220",
    "Implementing Party":"286",
    "Indicator":"308",
    "Individual":"500",
    "LogFrame Level 1":"300",
    "LogFrame Level 2":"301",
    "LogFrame Level 3":"302",
    "LogFrame Level 4":"303",
    "LogFrame Level 5":"304",
    "LogFrame Level 6":"305",
    "LogFrame Level 7":"306",
    "LogFrame Level 8":"307",
    "Marital Status":"506",
    "Month":"321",
    "Nationality":"504",
    "Partner":"295",
    "Partners":"266",
    "Payment":"430",
    "Planned Job Title":"270",
    "Procurement Line Item":"258",
    "Procurement Procedure":"264",
    "Procurement Required?":"217",
    "Procurement SOF":"263",
    "Project":"200",
    "Project Actions":"288",
    "Project DRC":"417",
    "Project Status":"201",
    "Quarter":"323",
    "Reason For Waiver":"268",
    "Region":"202",
    "Registration Country":"508",
    "Registration Region":"507",
    "Registration Sub-Division 1":"509",
    "Registration Sub-Division 2":"510",
    "Registration Sub-Division 3":"511",
    "Related Project":"274",
    "SCA Humanitarian Flexible Funding?":"213",
    "Semester":"322",
    "SOF":"400",
    "SOF DRC":"427",
    "Source of Action":"287",
    "Source of Fund (SOF)":"211",
    "Sub-Division 1":"206",
    "Sub-Division 2":"207",
    "Sub-Division 3":"208",
    "Task":"254",
    "Thematic Activity":"281",
    "Unit":"262",
    "Unit Type":"277",
    "Vulnerability":"505",
    "Waiver Required":"265",
    "Week":"324",
    "Year":"320",

    "Accountability to Communities":"2199",
    "Achievement":"3105",
    "Action Comments":"2192",
    "Actual":"3101",
    "Additional Notes":"2203",
    "Alias":"5051",
    "Association Address":"8029",
    "Association Code":"8006",
    "Association Estimated Size":"8030",
    "Association Name":"8000",
    "Baseline":"3103",
    "Budgeted Amount":"2220",
    "Budget Item Name":"2136",
    "Community Address":"7029",
    "Community Code":"7006",
    "Community Estimated Size":"7030",
    "Community Name":"7000",
    "Completion Date":"2187",
    "Cost":"2154",
    "Date of Birth":"5006",
    "Date of Issuance":"2184",
    "Date Purchase Request Issued":"2101",
    "Date the Purchase Request is Required":"2100",
    "Date to Start Recruitment":"2122",
    "Deadline for Action":"2185",
    "Delivery Date":"2106",
    " Description of Donor Requirement":"2131",
    "Donor Requires Approval to Changes":"2156",
    "Estimated Cost":"4059",
    "Estimated Cost":"2089",
    "Estimated Total Cost":"2091",
    "Estimated Transport Cost":"2090",
    "Ethnicity":"5015",
    "Evaluation and Research":"2200",
    "Household Address":"6019",
    "Household Code":"6018",
    "Household Estimated Size":"6017",
    "Household Name":"6000",
    "HR Plan ID Name":"2111",
    "Indicator Progress":"3106",
    "Indicator Progress(%)":"3107",
    "Individual Name":"5000",
    "Information Systems Management":"2201",
    "Issue that Requires Action":"2191",
    "Key Responsibilities":"2115",
    "LOE (%)":"2147",
    "Mobile Number":"5012",
    "Monitoring of Activities and Outputs":"2196",
    "Monitoring of Project Outcomes":"2197",
    "Monitoring Program Quality":"2198",
    "Notes":"5018",
    "# of Activities":"2036",
    "# of Associations":"8001",
    "# of Communities":"7001",
    "# of Households":"6001",
    "# of Individuals":"5001",
    "# of Projects":"2001",
    "# of SOFs":"4001",
    "# of Tasks":"2059",
    "Other Source of Action":"2186",
    "Partner Comments":"2226",
    "Passport No":"5054",
    "Payment Amount":"2223",
    "Planned Financial Report Date":"2225",
    "Planned Payment Date":"2224",
    "Procurement Line Item Name":"2077",
    "Procurement not executed by Supply Chain Team":"2155",
    "Project Name":"2000",
    "Quantity / Duration":"2145",
    "Recruitment Complete?":"2158",
    "Recruitment Required?":"2157",
    "Registration Address":"5029",
    "Remarks":"2108",
    "Remotely Managed Project?":"2123",
    "Required Delivery Date of Good/Service":"2098",
    "Role requires donor approval":"2159",
    "Service End Date":"2166",
    "Service Start Date":"2165",
    "Settlement Rural":"2031",
    "Settlement Urban":"2029",
    "Social Security National ID":"5011",
    "Source Details":"2188",
    "Special Support Needs":"6016",
    "Target":"3102",
    "Team Member Name":"2202",
    "Team Organogram":"2195",
    "Title":"2058",
    "Total Cost":"2149",
    "UNHCR No":"5055",


    "Project Action Name":"2179",
    "Start Date":"2021",
    "End Date":"2022",
    "Activity Progress (%)":"2133",
    "Action Status Name" : "2171",
    "Implementing Party Name": "2175",
    "Source of Action Name":"2177",
    "Actual Amount":"59230",
    "Planned Amount":"59231",
    "Duration":"1810",
    "Person Responsible":"1808",
    # "Award":"",
    # "Total Award Amount":"",
    "Status name":"2002",
    "Description":"1",
    "Created By":"2",
    "Created On":"3",
    "Modified By":"4",
    "Modified On":"5",
    "Task Progress (%)":"6"
}
ALL_OBJECTS_LIST = list(ALL_OBJECTS.values())


# MAX_WORD_LENGTH = 18
# MIN_WORD_LENGTH = 4
# MIN_PREDICTION_LENGTH=3
# LETTER_COUNT = 26
X_SHAPE = (ROW_MAX_PATTERN_LENGTH+COL_MAX_PATTERN_LENGTH, len(ALL_OBJECTS_LIST))
Y_SHAPE = (len(ALL_OBJECTS_LIST))
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

# def letterToFeatureIndex(letter):
#     return ord(letter) - 97#ord('a')

# def wordToData(letters):
#     dataX = numpy.zeros(X_SHAPE)
#     for i, letter in enumerate(letters):
#         dataX[i, letterToFeatureIndex(letter)] = 1
#     return dataX;

# def generateTrainingItem(letters, prediction):
#     dataX = wordToData(letters)
#     dataY = numpy.zeros(Y_SHAPE)
    
#     dataY[letterToFeatureIndex(prediction)] = 1
#     return dataX, dataY


# def genData2(words):
#     count = len(words)
#     x = [];
#     y = [];
#     for word in words:
#         for i in range(MIN_PREDICTION_LENGTH, len(word)-1, 1):
#             # print(word, word[:i], word[i])
#             dataX, dataY = generateTrainingItem(word[:i], word[i])
#             # print(dataX)
#             # print(dataY)
#             # print (word[:i])
#             # print(word[i+1])
#             x.append(dataX)
#             y.append(dataY);

#     x = numpy.asarray(x)
#     y = numpy.asarray(y)
#     return x, y

# def genData(size, f):
#     m = size;
#     x = numpy.zeros((m, 1));
#     y = numpy.zeros((m, 1));

#     for i in range(m):
#         xi = random.uniform(-10.0,10.0);
#         yi = f(xi)  
#         x[i]=xi;
#         y[i] = yi   
#     return x, y


def loadData():
    patterns = loadPatterns();
    X_data = []
    Y_data = []

    for pattern in patterns:#-2:-1
        rowPattern = pattern[1];
        if rowPattern=="":
            rowPattern = []
        else:
            rowPattern = rowPattern.split(",")
        
        columnPattern = pattern[2];
        if columnPattern=="":
            columnPattern = []
        else:
            columnPattern = columnPattern.split(",")
               
        rowPatternOneHot = make_one_hot([rowPattern])
        columnPatternOneHot = make_one_hot([columnPattern])
        couples = make_couples(rowPatternOneHot, columnPatternOneHot, ROW_MAX_PATTERN_LENGTH, COL_MAX_PATTERN_LENGTH)

        # x = numpy.zeros(X_SHAPE)
        # y = numpy.zeros(Y_SHAPE)

        
        X_couples = [[element[0][:-1] , element[1][:-1]] for element in couples]
        Y_couples = [[element[0][-1:] , element[1][-1:]] for element in couples]
        for x_couple, y_couple in zip(X_couples, Y_couples):
            x_couple[0] += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(ROW_MAX_PATTERN_LENGTH-len(x_couple[0]))]
            x_couple[1] += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
                for _ in range(COL_MAX_PATTERN_LENGTH-len(x_couple[1]))]
            
            
            # y_couple += [numpy.zeros(len(ALL_OBJECTS_LIST), dtype='int') 
            #     for _ in range(ROW_MAX_PATTERN_LENGTH-len(y_couple))]
            # if(len(x_couple[0] + x_couple[1])==16):
            #     print(recover(x_couple[0] + x_couple[1]))
            X_data.append(x_couple[0] + x_couple[1])
            Y_data.append(y_couple[0] + y_couple[1])
            # print(recover(y_couple[0] + y_couple[1]))

        # if(len(X_couples)==0):
        #     continue

        # X_data +=X_couples
        # Y_data+=Y_couples


    X_data = numpy.asarray(X_data)
    Y_data = numpy.asarray(Y_data)

    # print(X_data.shape)
    print(Y_data.shape)
    # for x,y in zip(X_data[0:100], Y_data[0:100]):
    #     print("{} / {} ".format(recover(x), recover(y)))
    # x_train, y_train = genData2(words)
    # x_test, y_test = x_train[0:100,:,:], y_train[0:100,:]
    x_cv, y_cv = numpy.expand_dims(numpy.zeros(X_SHAPE), 0), numpy.expand_dims(numpy.zeros(Y_SHAPE), 0) 
    return X_data, Y_data, X_data[0:1], Y_data[0:1], x_cv, y_cv
    # return x_train, y_train, x_test, y_test, x_cv, y_cv


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


def insertPatterToDB(reportid, rowCategories, columnCategories):
    now = datetime.datetime.now()
    with sqlite3.connect('/ml/patterns.sqlite') as conn:
        c = conn.cursor()
        c.execute('insert into patterns (reportid, rowPattern, columnPattern, datetime) values(?,?,?,?)', [reportid, rowCategories, columnCategories, now.strftime("%Y-%m-%d %H:%M")])
        conn.commit()

def loadPatterns():
    with sqlite3.connect('/ml/patterns.sqlite') as conn:
        c = conn.cursor()
        patterns = c.execute('SELECT * FROM patterns').fetchall()
        return patterns

def md(rowObjects, columnObjects):
    print(rowObjects, columnObjects)
    rowCategoriesIds = []
    for rowObject in rowObjects:
        rowCategoriesIds.append(nameToId(rowObject))

    columnCategoriesIds = []
    for columnObject in columnObjects:
        columnCategoriesIds.append(nameToId(columnObject))

    insertPatterToDB(-1, ','.join(rowCategoriesIds), ','.join(columnCategoriesIds))


def nameToId(objectName):
    if objectName in ALL_OBJECTS:
        return ALL_OBJECTS[objectName]
    else:
        raise Exception("Invalid entry {}".format(objectName));



def make_one_hot(patterns):
    patterns_one_hot = []
    for i, pattern in enumerate(patterns):
        # print(pattern)
        pattern_one_hot = []
        for j, obj in enumerate(pattern):
            vec = [0 for _ in range(len(ALL_OBJECTS_LIST))]
            vec[ALL_OBJECTS_LIST.index(obj)] = 1
            pattern_one_hot.append(vec)
        patterns_one_hot.append(pattern_one_hot)
    return patterns_one_hot


def generate_splits(pattern, max_pattern_length, weight=10):
    result = [pattern[:i] for i in range(2, len(pattern)+1)] * weight
    for i in range(1, len(pattern)-1):
        result += [pattern[i:j] for j in range(i+2, len(pattern)+1)]
    # for vec in result:
    #     vec += [numpy.zeros(len(vec[0]), dtype='int') 
    #             for _ in range(max_pattern_length-len(vec))]
    return result


def populateData(patterns, max_pattern_length):
    result_data = []
    for pattern in patterns:
        result_data += generate_splits(pattern, max_pattern_length)
    # return numpy.array(result_data)
    return result_data


def make_couples(row_patterns, col_patterns, row_max_pattern_length, col_max_pattern_length):
    row_populated_data = populateData(row_patterns, row_max_pattern_length)
    col_populated_data = populateData(col_patterns, col_max_pattern_length)
    
    couple_data = []
    for row_pattern in row_populated_data:
        for col_pattern in col_populated_data:
            couple_data.append([row_pattern, col_pattern])
    return couple_data
    # return numpy.array(couple_data)

def recover(pattern_one_hot):
    pattern = []
    for obj_vec in pattern_one_hot:
        if max(obj_vec)==0:
            pattern.append('-')
        else:
            pattern.append(ALL_OBJECTS_LIST[numpy.argmax(obj_vec)])
    return pattern




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
        insertPatterToDB(reportid, rowCategories, columnCategories)
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
    loadData()
    # md(["LogFrame Level 1", "Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Indicator"],["Actual", "Target", "Achievement"])
    # md(["Project", "Activity"],["Start Date", "End Date", "Activity Progress (%)"])
    # md(["Activity"],["Cost"])
    # md(["Project", "Activity"],["Start Date", "End Date", "Activity Progress (%)"])
    # md(["Estimated Thematic Area", "Project"],["Total Cost", "Project Action Name", "Action Status Name", "Implementing Party Name", "Source of Action Name", "Action Comments"])
    # md(["Project"],["Actual Amount", "Planned Amount", "# of Activities", "# of Tasks"])
    # md(["Project", "Indicator"],["Actual", "Target"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)", "Duration", "Person Responsible"])
    # # md(["Award"],["Total Award Amount"])
    # md(["Project"],["Status name"])
    # md(["Activity","Task"], ["Person Responsible", "Person Responsible", "Person Responsible", "Person Responsible"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)"])
    # md(["Project Actions"], ["Completion Date"])



    # md(["Project", "Activity", "Task"],["# of Activities", "# of Tasks", "Planned Amount", "Actual Amount"])
    # md(["Activity", "Indicator"],["Actual", "Target", "Achievement"])
    # md(["Project", "Activity"],["# of Tasks", "Planned Amount", "Actual Amount", "Start Date", "Description"])
    # md(["Project", "Activity"],["Actual Amount", "Planned Amount", "# of Activities", "# of Tasks"])
    # md(["Activity", "Task"],["Start Date", "End Date", "Task Progress (%)", "Duration", "Person Responsible"])
    # md(["Indicator"],["Baseline", "Actual", "Target", "Achievement"])
    # md(["Project", "Activity", "Budget Item"],["Created By", "Created On", "Modified By", "Modified On"])
    # md(["Activity","Task"], ["Person Responsible", "Person Responsible", "Person Responsible", "Person Responsible"])
    # md(["Project", "Activity", "Task"],["# of Activities", "# of Tasks", "Planned Amount", "Actual Amount"])


    # md(["Project", "LogFrame Level 1", "Indicator", "LogFrame Level 2", "Indicator", "LogFrame Level 3", "Indicator" ],["Baseline", "Actual", "Target", "Achievement"])

    # md(["LogFrame Level 1", "Indicator", "LogFrame Level 2", "Indicator"],[])
    # md(["Project", "Activity"], ["Payment Amount", "Budgeted Amount"])


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











# md(["Indicator", "Disability"],["Total Cost", "Passport No"])


