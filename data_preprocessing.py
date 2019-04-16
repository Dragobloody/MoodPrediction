import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/dataset_mood_smartphone.csv'


#################################### LOAD DATA #########################################

data = pd.read_csv(DATA_PATH, encoding = 'utf-8')
data = data.drop(columns = data.columns[0])
pacient_ID = data.columns[0]
time = data.columns[1]
attribute = data.columns[2]
value = data.columns[3]

data = data.sort_values(by = [pacient_ID,time,attribute])
data[time] = (pd.to_datetime(data[time], format='%Y-%m-%d %H:%M:%S.%f')).dt.date
data = data.dropna()

ids = data[pacient_ID].unique()
variables = data[attribute].unique()
dates =  data[time].unique()   


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



###################### REMOVE DAYS WITH NO RELEVANT DATA(SPARSE DATA) ######################### 

data = pd.crosstab(index = [data['id'],data['time']], 
                   columns = data['variable'], values = data['value'], 
                   aggfunc = lambda x:x).reset_index()


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





def init_pacients(data1):
    pacients = {}
    for pid in ids:
        pacients[pid] = {}
        pacient_dataframe = data1.loc[data1[pacient_ID] == pid]
        pacient_dates = pacient_dataframe[time].unique()
        
        for date in pacient_dates:
            pacients[pid][date] = {}
            for var in variables:
                pacients[pid][date][var] = float('nan')
    for idx, row in data1.iterrows():    
        pacients[row['id']][row['time']][row['variable']] = row['value']
    return pacients
            

pacients = init_pacients(data1)

def plots(data1,variables):    
    for var in variables:
        values = list(data1.loc[data1['variable'] == var]['value'])
        plt.figure()
        plt.hist(values)        
        plt.savefig('plots/hist_' + var + '.png')
        

plots(data_5,variables)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


            
   
    
    
    