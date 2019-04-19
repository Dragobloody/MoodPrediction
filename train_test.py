import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################ Training/Testing Data ############################
    
def train_test_individual_pacient(patient_data, mood_pacient_data, ids, ratio_train, period = 3, ravel=True):
    ''' all variables are patient specific
        patient _data - dictionar with data for each patient
        mood_pacient_data - same data with original mood (not standardized)
        ratio_train - ration from the original data to use for train
        period - how many days to use for regression
    '''
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for pid in ids:
        split_index = int(patient_data[pid].shape[0]*ratio_train)
        pid_data = patient_data[pid]
        mood_pid_data = mood_pacient_data[pid]
        
        ## Train dataset
        X_train[pid] = []
        y_train[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index - period + 1):
            x = pid_data.iloc[i:i+period, :].values
            if ravel:
                x = x.ravel(order='F')
            X_train[pid].append(x)
            y_train[pid].append(mood_pid_data.iloc[i+period].mood)
        
        X_train[pid] = np.array(X_train[pid])
        y_train[pid] = np.array(y_train[pid])
            
        ## Test dataset
        X_test[pid] = []
        y_test[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index, pid_data.shape[0] - period):
            x = pid_data.iloc[i:i+period, :].values
            if ravel:
                x = x.ravel(order='F')
            X_test[pid].append(x)
            y_test[pid].append(mood_pid_data.iloc[i+period].mood)
            
            
        X_test[pid] = np.array(X_test[pid])
        y_test[pid] = np.array(y_test[pid])
    
    return (X_train, y_train, X_test, y_test)


def train_test_pca(patient_data, mood_pacient_data, ids, ratio_train, period = 3, ravel=True):
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for pid in ids:
        split_index = int(patient_data[pid].shape[0]*ratio_train)
        pid_data = patient_data[pid]
        mood_pid_data = mood_pacient_data[pid]
        
        ## Train dataset
        X_train[pid] = []
        y_train[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index - period + 1):
            x = pid_data[i:i+period, :]*(np.array([0.1,0.3,0.6])[:,np.newaxis])
            if ravel:
                x = x.ravel(order='F')
            X_train[pid].append(x)
            y_train[pid].append(mood_pid_data.iloc[i+period].mood)
        
        X_train[pid] = np.array(X_train[pid])
        y_train[pid] = np.array(y_train[pid])
            
        ## Test dataset
        X_test[pid] = []
        y_test[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index, pid_data.shape[0] - period):
            x = pid_data[i:i+period, :]*(np.array([0.1,0.3,0.6])[:,np.newaxis])
            if ravel:
                x = x.ravel(order='F')
            X_test[pid].append(x)
            y_test[pid].append(mood_pid_data.iloc[i+period].mood)
            
            
        X_test[pid] = np.array(X_test[pid])
        y_test[pid] = np.array(y_test[pid])
    
    return (X_train, y_train, X_test, y_test)


def train_test_all(X_train, Y_train, X_test, Y_test, ids):
    X_train_all = np.concatenate([X_train[pid] for pid in ids], axis=0)
    Y_train_all = np.concatenate([Y_train[pid] for pid in ids], axis=0)
    X_test_all = np.concatenate([X_test[pid] for pid in ids], axis=0)
    Y_test_all = np.concatenate([Y_test[pid] for pid in ids], axis=0)
    
    return X_train_all, Y_train_all, X_test_all, Y_test_all

################################# Evaluation function ###############################
#Issue #1

def mse(prediction, target):
    return np.sum((target - prediction)**2)/target.size
        
#################################### Benchmark alg ######################################
# Issue #2
    
def benchmark(X_test, ids, period):
    predictions = {}
    all_predictions = []
    aux = 0
    for pid in ids:        
        predictions[pid] = (X_test[pid])[:,period - 1]
        
    all_predictions = np.concatenate([predictions[pid] for pid in ids], axis=0)
    
    predictions['all'] = all_predictions        
    return predictions


#################################### SVM ######################################
# Issue #3
    
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate, cross_val_score


def choose_algorithm(X, y):
    algorithms = [
        SVR(kernel='rbf', gamma = 'auto', C=1),
        SVR(kernel='rbf', gamma = 'scale', C=1),
        SVR(kernel='poly', degree=2, gamma = 'auto', C=1),
        SVR(kernel='poly', degree=3, gamma = 'auto', C=1),
        SVR(kernel='poly', degree=5, gamma = 'auto', C=1),
        SVR(kernel='poly', degree=7, gamma = 'auto', C=1),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=0, loss='lad'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, max_depth=3, random_state=0, loss='lad'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=3, random_state=0, loss='lad'),
        MLPRegressor(hidden_layer_sizes=[50,50,50],solver='lbfgs',learning_rate='adaptive'),
        MLPRegressor(hidden_layer_sizes=[100,100],solver='lbfgs',learning_rate='adaptive'),
        MLPRegressor(hidden_layer_sizes=[30,30,30,30,30],solver='lbfgs',learning_rate='adaptive'),
        DecisionTreeRegressor(max_depth=3)
    ]
    scores = []
    for algorithm in algorithms:
        cv = cross_val_score(algorithm, X, y, cv=10, scoring='neg_mean_squared_error')
        scores += [sum(cv)/len(cv)]
    best = (np.array(scores)).argmax(0)
    return algorithms[best]

def regression(algorithm,
               X_train, y_train, X_test,
               X_train_all, Y_train_all, X_test_all,
               ids):
    predictions = {}
    algorithm.fit(X_train_all, Y_train_all)
    predictions['all'] = algorithm.predict(X_test_all)
    for pid in ids:
        algorithm.fit(X_train[pid], y_train[pid])
        predictions[pid] = algorithm.predict(X_test[pid])
         
    return predictions
        
        

#################################### ARIMA ######################################
#TODO Issue #4

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error

def lstm(X_train_all, Y_train_all, X_test_all, Y_test_all):
    model = Sequential()
    model.add(LSTM(10, input_shape=(X_train_all.shape[1], X_train_all.shape[2]), return_sequences=True))
    # model.add(Dropout(rate=0.2))
    model.add(LSTM(10))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(1))
    optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=3e-4, nesterov=True)
    # optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train_all, Y_train_all, epochs=200, batch_size=32, verbose=2,
              validation_data=(X_test_all, Y_test_all),
              callbacks=[reduce_lr])
    return model.evaluate(X_test_all, Y_test_all)
 

#################################### result Statistics ######################################
# Issue #5

def prediction_stats(predictions, y_test, y_test_all, ids, eval_func = mse):
    result = []
    result_all = eval_func(predictions['all'], y_test_all)
    
    for pid in ids:
        result.append(eval_func(predictions[pid], y_test[pid]))
        
    return np.array(result), result_all
    




