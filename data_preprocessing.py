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
        pacient_dataframe = data.loc[data.id == pid]
        pacient_dates = pacient_dataframe['time'].unique()
        
        media = {}
        media['mood'] = (pacient_dataframe.loc[pacient_dataframe.variable == 'mood'])['value'].mean()
        media['activity'] = (pacient_dataframe.loc[pacient_dataframe.variable == 'activity'])['value'].mean()
        media['circumplex.arousal'] = (pacient_dataframe.loc[pacient_dataframe.variable == 'circumplex.arousal'])['value'].mean()
        media['circumplex.valence'] = (pacient_dataframe.loc[pacient_dataframe.variable == 'circumplex.valence'])['value'].mean()
        
        for day in pacient_dates:
            pacient_day_dataframe = pacient_dataframe.loc[pacient_dataframe['time'] == day]
            day_variables = pacient_day_dataframe['variable'].unique()
            
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
        pacient_dataframe = data.loc[data.id == pid]
        pacient_dates = pacient_dataframe['time'].unique()       
        for day in pacient_dates:            
            if next(data.loc[(data.id == pid) & (data.time == day)].iterrows())[1].screen == 0:                
                data = data.drop(data[(data.id == pid) & (data.time == day)].index)
            else:
                break
            
    return data



################################## CORRELATION MATRXIX ####################################
def correlation(data,ids):
    data_corr = data.drop(columns = ['id','time'])
    total_corr = data_corr.corr()
    pacient_mood_corr = pd.DataFrame()
    
    pacient_corr= {}    
    for pid in ids:
        pacient_dataframe = data.loc[data.id == pid].drop(columns = ['id','time'])
        pacient_corr[pid] = pacient_dataframe.corr()
        pacient_mood_corr[pid] = pacient_corr[pid].mood 
        
    
    return total_corr, pacient_corr, pacient_mood_corr


####################### INDIVIDUALIZE DATA AND REMOVE INDIVIDUAL DIMS ######################
def individualize(data,ids, pacient_mood_corr):
    pacients = {}  
    pacients['all'] = data.drop(columns = 'screen')
    for pid in ids:
         pacient_dataframe = data.loc[data.id == pid]
         pacient_corr = pacient_mood_corr[pid]
         nan_vars = pacient_corr.index[(pacient_corr.isna())].tolist()
         nan_vars += ['id','screen','time']                                       
         
         pacient_dataframe= pacient_dataframe.drop(columns = nan_vars)
         pacients[pid] = pacient_dataframe
    
    return pacients



#################################### STANDARDIZATION #########################################
mean_var = set(['mood','activity','circumplex.arousal','circumplex.valence'])

def standardization(pacients_original, ids, mean_var):
    pacients_standardized = {}
    scaled_features = pacients_original['all'].copy()
    col_names = set(scaled_features.columns)     
    col_names = list(col_names-mean_var)
        
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)    
    scaled_features[col_names] = features  
        
    pacients_standardized['all'] = scaled_features
    for pid in ids:        
        scaled_features = pacients_original[pid].copy()
        col_names = set(scaled_features.columns)        
        col_names = list(col_names-mean_var)
        
        features = scaled_features[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)    
        scaled_features[col_names] = features  
        
        pacients_standardized[pid] = scaled_features
    
    return pacients_standardized


######################################## PCA ##############################################
    
def pca(pacients_standardized, ids):
    pacients_pca = {}   
    for pid in ids:        
        pca = PCA(n_components=10)
        pacients_pca[pid] = pca.fit_transform(pacients_standardized[pid])
    return pacients_pca


    
    
    

        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


            
   
    
    
    