import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################ Training/Testing Data ############################
    
def train_test_individual_patient(patient_data, mood_patient_data, ids, ratio_train, period = 3, ravel=True):
    ''' all variables are patient specific
        patient _data - dictionar with data for each patient
        mood_patient_data - same data with original mood (not standardized)
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
        mood_pid_data = mood_patient_data[pid]
        
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


def train_test_pca(patient_data, mood_patient_data, ids, ratio_train, period = 3, ravel=True):
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for pid in ids:
        split_index = int(patient_data[pid].shape[0]*ratio_train)
        pid_data = patient_data[pid]
        mood_pid_data = mood_patient_data[pid]
        
        ## Train dataset
        X_train[pid] = []
        y_train[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index - period + 1):
            x = pid_data[i:i+period, :]*(np.linspace(0.1,0.9,period)[:,np.newaxis])
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
            x = pid_data[i:i+period, :]*(np.linspace(0.1,0.9,period)[:,np.newaxis])
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


def train_test(patient_data, labels, ids, ratio_train = 0.6, period = 3):
    ''' Split into test and train data
        all variables are patient specific
        patient _data - dictionar with data for each patient
        labels - labels(mood)
        ratio_train - ration from the original data to use for train
        period - how many days to use for regression
    '''
    X_train = {}
    Y_train = {}
    X_test = {}
    Y_test = {}
    for pid in ids:
        pid_data = patient_data[pid]
        split_index = int(pid_data.shape[0]*ratio_train)
        X_train[pid] = pid_data.iloc[0:split_index, :].values
        Y_train[pid] = pid_data.iloc[0:split_index, :].mood.values
        X_test[pid] = pid_data.iloc[split_index:, :].values
        Y_test[pid] = pid_data.iloc[split_index:, :].mood.values    
    
    return X_train, Y_train, X_test, Y_test


def make_sequences(X_train, X_test, moods_train, moods_test, period=3):
    X_train = {k: np.array([X_train[k][i:i+period, :] for i in range(X_train[k].shape[0]-period)]) for k in X_train.keys()}
    Y_train = {k: np.array([moods_train[k][i+period, 0] for i in range(moods_train[k].shape[0]-period)]) for k in X_train.keys()}
    
    X_test = {k: np.array([X_test[k][i:i+period, :] for i in range(X_test[k].shape[0]-period)]) for k in X_test.keys()}
    Y_test = {k: np.array([moods_test[k][i+period, 0] for i in range(moods_test[k].shape[0]-period)]) for k in X_test.keys()}
    
    return X_train, Y_train, X_test, Y_test


def average_timesteps(data):
    """ Take average over timesteps for each patient """
    return {k: np.mean(data[k], axis=1) for k in data.keys()}


def flatten_timesteps(data):
    """ Flatten timesteps for each patient """
    return {k: data[k].reshape((data[k].shape[0], -1)) for k in data.keys()}
    
    
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
        predictions[pid] = (X_test[pid])[period-1:-1,0]
        
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
#         SVR(kernel='poly', degree=2, gamma = 'auto', C=1),
#         SVR(kernel='poly', degree=3, gamma = 'auto', C=1),
#         SVR(kernel='poly', degree=5, gamma = 'auto', C=1),
#         SVR(kernel='poly', degree=7, gamma = 'auto', C=1),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=0, loss='lad'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, max_depth=3, random_state=0, loss='lad'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=3, random_state=0, loss='ls'),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=3, random_state=0, loss='lad'),
#         MLPRegressor(hidden_layer_sizes=[50,50,50],solver='lbfgs',learning_rate='adaptive'),
#         MLPRegressor(hidden_layer_sizes=[100,100],solver='lbfgs',learning_rate='adaptive'),
#         MLPRegressor(hidden_layer_sizes=[30,30,30,30,30],solver='lbfgs',learning_rate='adaptive'),
        DecisionTreeRegressor(max_depth=3)
    ]
    scores = []
    for algorithm in algorithms:
        cv = cross_val_score(algorithm, X, y, cv=10, scoring='neg_mean_squared_error')
        print(algorithm)
        print(sum(cv)/len(cv))
        scores += [sum(cv)/len(cv)]
    best = (np.array(scores)).argmax(0)
    return algorithms[best]

def regression(algorithm,
               X_train, y_train, X_test,
               X_train_all, Y_train_all, X_test_all,
               ids, per_patient_only=False):
    predictions = {}
    if not per_patient_only:
        algorithm.fit(X_train_all, Y_train_all)
        predictions['all'] = algorithm.predict(X_test_all)
    for pid in ids:
        algorithm.fit(X_train[pid], y_train[pid])
        predictions[pid] = algorithm.predict(X_test[pid])
         
    return predictions
        
        
#################################### result Statistics ######################################

def prediction_stats(predictions, y_test, y_test_all, ids, eval_func = mse, per_patient_only=False):
    result = []
    result_all = eval_func(predictions['all'], y_test_all) if not per_patient_only else 0
    
    for pid in ids:
        result.append(eval_func(predictions[pid], y_test[pid]))
        
    return np.array(result), result_all
    




