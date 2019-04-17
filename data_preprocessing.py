import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'data/dataset_mood_smartphone.csv'


#################################### LOAD DATA #########################################

data = pd.read_csv(DATA_PATH, encoding = 'utf-8')
data = data.drop(columns = data.columns[0])
pacient_ID = data.columns[0]
time = data.columns[1]
attribute = data.columns[2]
value = data.columns[3]

# sort data in DataFrame
data = data.sort_values(by = [pacient_ID,time,attribute])
# get rig of time leave only date
data[time] = (pd.to_datetime(data[time], format='%Y-%m-%d %H:%M:%S.%f')).dt.date
# get rid of NaN rows (occurs only for circumplex.*)
data = data.dropna()

ids = data[pacient_ID].unique()
variables = data[attribute].unique()
 


############################ GET STATISTICS AND CLEAN DATA ###############################

def stats(data, variables):
    final_stats = pd.DataFrame()
    data_stats = data.drop(columns = [data.columns[0],data.columns[1]])
    for var in variables:
        var_dataframe = data_stats.loc[data_stats[attribute] == var]
        stats = var_dataframe.describe(percentiles = [0.05,0.1, 0.25, 0.5, 0.75, 0.95])       
        final_stats[var] = stats['value']
    return final_stats.transpose()


def remove_5(data):
    data_new = data
    low = 0.05
    high = 0.95
    data_stats = data.drop(columns = [data.columns[0],data.columns[1]])   
    for var in variables[6:]:       
        var_dataframe = data_stats.loc[data_stats[attribute] == var]
        var_quant = var_dataframe.drop(columns = var_dataframe.columns[0]).quantile([low,high])       
    
        data_new = data_new.drop(data_new[(data_new[attribute] == var) &
                     ((data_new.value < var_quant.loc[0.05,'value']) |
                     (data_new.value > var_quant.loc[0.95,'value']))].index)
    
    return data_new

data = remove_5(data)
table = stats(data, variables)
table2 = stats(data, variables) 



########################## GROUP DATA BY PACIENT, DAY, VARIABLE ############################### 
    
data_call_sms = data.loc[~data[attribute].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby([pacient_ID,time,attribute]).sum().reset_index()
data_else = data.loc[data[attribute].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby([pacient_ID,time,attribute]).mean().reset_index()
data = pd.concat([data_call_sms,data_else]).sort_values(by = [pacient_ID,time,attribute])


################################ DEALING WITH MISSING DATA ################################### 

def fill_missing_data(data, ids, variables):
    mean_var = set(['mood','activity','circumplex.arousal','circumplex.valence'])
    zero_var = set(variables) - mean_var
    for pid in ids:        
        pacient_dataframe = data.loc[data[pacient_ID] == pid]
        pacient_dates = pacient_dataframe[time].unique()
        
        media = {}
        media['mood'] = (pacient_dataframe.loc[pacient_dataframe[attribute] == 'mood'])['value'].mean()
        media['activity'] = (pacient_dataframe.loc[pacient_dataframe[attribute] == 'activity'])['value'].mean()
        media['circumplex.arousal'] = (pacient_dataframe.loc[pacient_dataframe[attribute] == 'circumplex.arousal'])['value'].mean()
        media['circumplex.valence'] = (pacient_dataframe.loc[pacient_dataframe[attribute] == 'circumplex.valence'])['value'].mean()
        
        for day in pacient_dates:
            pacient_day_dataframe = pacient_dataframe.loc[pacient_dataframe[time] == day]
            day_variables = pacient_day_dataframe[attribute].unique()
            
            missing_variables = set(variables) - set(day_variables)
            mean_variables = missing_variables.intersection(mean_var)
            zero_variables = missing_variables.intersection(zero_var)
            
            for var in zero_variables:
                data = data.append({'id':pid,'time':day,'variable':var,'value':0},ignore_index=True)
            for var in mean_variables:
                data = data.append({'id':pid,'time':day,'variable':var,'value':media[var]},ignore_index=True)
    
    data = data.sort_values(by = [pacient_ID,time,attribute])
    return data

data =  fill_missing_data(data, ids, variables)
data.to_csv(path_or_buf = 'data/data.csv')

# data = pd.read_csv('data/data.csv')
# data = data.drop(columns = data.columns[0])

###################### REMOVE DAYS WITH NO RELEVANT DATA(SPARSE DATA) ######################### 

data = pd.crosstab(index = [data['id'],data['time']], 
                   columns = data['variable'], values = data['value'], 
                   aggfunc = lambda x:x).reset_index()

# Swap 3rd column with mood, so later mood is the first column
columns_new = data.columns.values.copy() 
columns_new[2], columns_new[18] = columns_new[18], columns_new[2]
data = data[columns_new]



def remove_days(data):
    for pid in ids:
        pacient_dataframe = data.loc[data[pacient_ID] == pid]
        pacient_dates = pacient_dataframe[time].unique()       
        for day in pacient_dates:            
            if next(data.loc[(data.id == pid) & (data.time == day)].iterrows())[1].screen == 0:                
                data = data.drop(data[(data.id == pid) & (data.time == day)].index)
            else:
                break
            
    return data

data = remove_days(data)
data.to_csv(path_or_buf = 'data/data.csv')



################################## CORRELATION MATRXIX ####################################

def correlation(data):
    data_corr = data.drop(columns = ['id','time'])
    total_corr = data_corr.corr()
    pacient_mood_corr = pd.DataFrame()
    
    pacient_corr= {}    
    for pid in ids:
        pacient_dataframe = data.loc[data[pacient_ID] == pid].drop(columns = ['id','time'])
        pacient_corr[pid] = pacient_dataframe.corr()
        pacient_mood_corr[pid] = pacient_corr[pid].mood 
        
    
    return total_corr, pacient_corr, pacient_mood_corr

total_corr, pacient_corr, pacient_mood_corr = correlation(data)


####################### INDIVIDUALIZE DATA AND REMOVE INDIVIDUAL DIMS ######################

def individualize(data, pacient_mood_corr):
    pacients = {}    
    for pid in ids:
         pacient_dataframe = data.loc[data[pacient_ID] == pid]
         pacient_corr = pacient_mood_corr[pid]
         nan_vars = pacient_corr.index[(pacient_corr.isna()) | 
                                       ((pacient_corr > -0.05) &
                                        (pacient_corr < 0.05))].tolist()
         nan_vars += ['id','screen']                                       
         
         pacient_dataframe= pacient_dataframe.drop(columns = nan_vars)
         pacients[pid] = pacient_dataframe
    
    return pacients

pacients_original = individualize(data, pacient_mood_corr)


#################################### STANDARDIZATION #########################################

pacients_standardized = {}
for pid in ids:    
    ss = StandardScaler()
    scaled_features = pacients_original[pid].drop(columns = 'time').copy()
    col_names = set(scaled_features.columns)
    mean_var = set(['mood','activity','circumplex.arousal','circumplex.valence'])
    col_names = list(col_names-mean_var)
    
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)    
    scaled_features[col_names] = features  
    
    pacients_standardized[pid] = scaled_features
    
    
    
################################ Training/Testing Data ############################
    
def train_test(patient_data, ratio_train, period = 3):
    ''' all variables are patient specific
        patient _data - dictionar with data for each patient
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
        
        ## Train dataset
        X_train[pid] = []
        y_train[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index - period + 1):
            X_train[pid].append(pid_data.iloc[i:i+period, :].values.ravel(order='F'))
            y_train[pid].append(pid_data.iloc[i+period].mood)
        
        X_train[pid] = np.array(X_train[pid])
        y_train[pid] = np.array(y_train[pid])
            
        ## Test dataset
        X_test[pid] = []
        y_test[pid] = []
        # first day from test is also the last label for train that is why we add 1
        for i in range(split_index, pid_data.shape[0] - period):
            X_test[pid].append(pid_data.iloc[i:i+period, :].values.ravel(order='F'))
            y_test[pid].append(pid_data.iloc[i+period].mood)
            
            
        X_test[pid] = np.array(X_test[pid])
        y_test[pid] = np.array(y_test[pid])
    
    return (X_train, y_train, X_test, y_test)

        
      
    
X_train, y_train, X_test, y_test = train_test(pacients_standardized, 0.75)


################################# Evaluation function ###############################
#Issue #1

def mse(prediction, target):
    return np.sum((target - prediction)**2)/target.size
        
#################################### Benchmark alg ######################################
# Issue #2
    
def benchmark(X_test, period):
    predictions = {}
    for pid in ids:
        predictions[pid] = (X_test[pid])[:,period - 1]
        
    return predictions


#################################### SVM ######################################
#TODO Issue #3

#################################### ARIMA ######################################
#TODO Issue #4

#################################### result Statistics ######################################
#TODO Issue #5



##############################  Tests ###########################
    
bench_predictions = benchmark(X_test, period = 3)




def plots(data1,variables):    
    for var in variables:
        values = list(data1.loc[data1['variable'] == var]['value'])
        plt.figure()
        plt.hist(values)        
        plt.savefig('plots/hist_' + var + '.png')
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


            
   
    
    
    