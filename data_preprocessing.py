import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#################################### LOAD DATA #########################################

def load_data(path):   
    data = pd.read_csv(path, encoding = 'utf-8')
    data = data.drop(columns = data.columns[0])    
    # sort data in DataFrame
    data = data.sort_values(by = ['id','time','variable'])
    # get rig of time leave only date
    data.time = (pd.to_datetime(data.time, format='%Y-%m-%d %H:%M:%S.%f')).dt.date
    # get rid of NaN rows (occurs only for circumplex.*)
    data = data.dropna()
    
    ids = data['id'].unique()
    variables = data['variable'].unique()
    
    return data, ids, variables


############################ GET STATISTICS AND CLEAN DATA ###############################

def stats(data, variables):
    final_stats = pd.DataFrame()
    data_stats = data.drop(columns = [data.columns[0],data.columns[1]])
    for var in variables:
        var_dataframe = data_stats.loc[data_stats.variable == var]
        stats = var_dataframe.describe(percentiles = [0.05,0.1, 0.25, 0.5, 0.75, 0.95])       
        final_stats[var] = stats['value']
    return final_stats.transpose()


def remove_5(data, variables):
    data_new = data
    low = 0.05
    high = 0.95
    data_stats = data.drop(columns = [data.columns[0],data.columns[1]])   
    for var in variables[6:]:       
        var_dataframe = data_stats.loc[data_stats.variable == var]
        var_quant = var_dataframe.drop(columns = var_dataframe.columns[0]).quantile([low,high])       
    
        data_new = data_new.drop(data_new[(data_new.variable == var) &
                     ((data_new.value < var_quant.loc[0.05,'value']) |
                     (data_new.value > var_quant.loc[0.95,'value']))].index)
    
    return data_new


################################ DEALING WITH MISSING DATA ################################### 

def fill_missing_data(data, ids, variables):
    mean_var = set(['mood','activity','circumplex.arousal','circumplex.valence'])
    zero_var = set(variables) - mean_var
    for pid in ids:        
        patient_dataframe = data.loc[data.id == pid]
        patient_dates = patient_dataframe['time'].unique()
        
        media = {}
        media['mood'] = (patient_dataframe.loc[patient_dataframe.variable == 'mood'])['value'].mean()
        media['activity'] = (patient_dataframe.loc[patient_dataframe.variable == 'activity'])['value'].mean()
        media['circumplex.arousal'] = (patient_dataframe.loc[patient_dataframe.variable == 'circumplex.arousal'])['value'].mean()
        media['circumplex.valence'] = (patient_dataframe.loc[patient_dataframe.variable == 'circumplex.valence'])['value'].mean()
        
        for day in patient_dates:
            patient_day_dataframe = patient_dataframe.loc[patient_dataframe['time'] == day]
            day_variables = patient_day_dataframe['variable'].unique()
            
            missing_variables = set(variables) - set(day_variables)
            mean_variables = missing_variables.intersection(mean_var)
            zero_variables = missing_variables.intersection(zero_var)
            
            for var in zero_variables:
                data = data.append({'id':pid,'time':day,'variable':var,'value':0},ignore_index=True)
            for var in mean_variables:
                data = data.append({'id':pid,'time':day,'variable':var,'value':media[var]},ignore_index=True)
    
    data = data.sort_values(by = ['id','time','variable'])
    return data



###################### REMOVE DAYS WITH NO RELEVANT DATA(SPARSE DATA) ######################### 


def remove_days(data,ids):
    for pid in ids:
        patient_dataframe = data.loc[data.id == pid]
        patient_dates = patient_dataframe['time'].unique()       
        for day in patient_dates:            
            if next(data.loc[(data.id == pid) & (data.time == day)].iterrows())[1].screen == 0:                
                data = data.drop(data[(data.id == pid) & (data.time == day)].index)
            else:
                break
            
    return data



################################## CORRELATION MATRXIX ####################################
def correlation(data,ids):
    data_corr = data.drop(columns = ['id','time'])
    total_corr = data_corr.corr()
    patient_mood_corr = pd.DataFrame()
    
    patient_corr= {}    
    for pid in ids:
        patient_dataframe = data.loc[data.id == pid].drop(columns = ['id','time'])
        patient_corr[pid] = patient_dataframe.corr()
        patient_mood_corr[pid] = patient_corr[pid].mood 
        
    
    return total_corr, patient_corr, patient_mood_corr


####################### INDIVIDUALIZE DATA AND REMOVE INDIVIDUAL DIMS ######################
def individualize(data,ids, patient_mood_corr):
    patients = {}     
    for pid in ids:
         patient_dataframe = data.loc[data.id == pid]
         patient_corr = patient_mood_corr[pid]
         nan_vars = patient_corr.index[(patient_corr.isna()) |  
                                       ((patient_corr > -0.01) & 
                                        (patient_corr < 0.01))].tolist()
         nan_vars += ['id','screen','time']                                       
         
         patient_dataframe= patient_dataframe.drop(columns = nan_vars)
         patients[pid] = patient_dataframe
    
    return patients


#################################### STANDARDIZATION #########################################

def standardization(X_train, X_test, ids):
    X_train_st = {}
    X_test_st = {}
    
    for pid in ids:        
        scaler = StandardScaler()
        scaler = scaler.fit(X_train[pid])
        X_train_st[pid] = scaler.transform(X_train[pid])
        X_test_st[pid] = scaler.transform(X_test[pid])
    
    return X_train_st, X_test_st

######################################## PCA ##############################################

def pca(X_train, X_test, ids):    
    X_train_pca = {}
    X_test_pca = {}
    for pid in ids:        
        pca = PCA(n_components=3)
        pca = pca.fit(X_train[pid])
        X_train_pca[pid] = pca.transform(X_train[pid])
        X_test_pca[pid] = pca.transform(X_test[pid])
    return X_train_pca, X_test_pca

