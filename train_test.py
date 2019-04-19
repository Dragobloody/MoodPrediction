import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################ Training/Testing Data ############################
    
def train_test_individual_pacient(patient_data, mood_pacient_data, ids, ratio_train, period = 3):
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
            X_train[pid].append(((pid_data.iloc[i:i+period, :].values)).ravel(order='F'))
            y_train[pid].append(mood_pid_data.iloc[i+period].mood)
        
        X_train[pid] = np.array(X_train[pid])
        y_train[pid] = np.array(y_train[pid])
            
        ## Test dataset
        X_test[pid] = []
        y_test[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index, pid_data.shape[0] - period):
            X_test[pid].append(((pid_data.iloc[i:i+period, :].values)).ravel(order='F'))
            y_test[pid].append(mood_pid_data.iloc[i+period].mood)
            
            
        X_test[pid] = np.array(X_test[pid])
        y_test[pid] = np.array(y_test[pid])
    
    return (X_train, y_train, X_test, y_test)


def train_test_pca(patient_data, mood_pacient_data, ids, ratio_train, period = 3):
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
            X_train[pid].append(((pid_data[i:i+period, :])*(np.array([0.1,0.3,0.6])[:,np.newaxis])).ravel(order='F'))
            y_train[pid].append(mood_pid_data.iloc[i+period].mood)
        
        X_train[pid] = np.array(X_train[pid])
        y_train[pid] = np.array(y_train[pid])
            
        ## Test dataset
        X_test[pid] = []
        y_test[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index, pid_data.shape[0] - period):
            X_test[pid].append(((pid_data[i:i+period, :])*(np.array([0.1,0.3,0.6])[:,np.newaxis])).ravel(order='F'))
            y_test[pid].append(mood_pid_data.iloc[i+period].mood)
            
            
        X_test[pid] = np.array(X_test[pid])
        y_test[pid] = np.array(y_test[pid])
    
    return (X_train, y_train, X_test, y_test)


def train_test_all(X_train, Y_train, X_test, Y_test, ids):
    X_train_all = []
    Y_train_all = []
    X_test_all = []
    Y_test_all = []
    aux = 0
    
    for pid in ids:
        if aux == 0:
            X_train_all = X_train[pid]
            Y_train_all = Y_train[pid]
            X_test_all = X_test[pid]
            Y_test_all = Y_test[pid]
            aux = 1
        else:
            X_train_all = np.concatenate((X_train_all,X_train[pid]), axis =0)
            Y_train_all = np.concatenate((Y_train_all,Y_train[pid]), axis =0)
            X_test_all = np.concatenate((X_test_all,X_test[pid]), axis =0)
            Y_test_all = np.concatenate((Y_test_all,Y_test[pid]), axis =0)
    
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
        
        if aux == 0:
            all_predictions =  predictions[pid]
            aux = 1
        else:
            all_predictions = np.concatenate((all_predictions,predictions[pid]), axis = 0)
    
    predictions['all'] = all_predictions        
    return predictions


#################################### SVM ######################################
# Issue #3
    
from sklearn.svm import SVR

def svm_regression(X_train, y_train, X_test, 
                   X_train_all, Y_train_all, X_test_all, 
                   ids, kernel = 'poly', degree = 3):
    predictions = {}
    svm_reg = SVR(kernel = kernel, degree = degree, gamma = 'auto')
    svm_reg.fit(X_train_all, Y_train_all)
    predictions['all'] = svm_reg.predict(X_test_all)
    for pid in ids:
        svm_reg = SVR(kernel = kernel, degree = degree, gamma = 'auto')
        svm_reg.fit(X_train[pid], y_train[pid])
        predictions[pid] = svm_reg.predict(X_test[pid])
        
    return predictions
        
        
        

#################################### ARIMA ######################################
#TODO Issue #4

#################################### result Statistics ######################################
# Issue #5

def prediction_stats(predictions, y_test, y_test_all, ids, eval_func = mse):
    result = []
    result_all = eval_func(predictions['all'], y_test_all)
    
    for pid in ids:
        result.append(eval_func(predictions[pid], y_test[pid]))
        
    return np.array(result), result_all
    




