import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/dataset_mood_smartphone.csv'


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
    
data_call_sms = data.loc[~data[attribute].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby([pacient_ID,time,attribute]).sum().reset_index()
data_else = data.loc[data[attribute].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby([pacient_ID,time,attribute]).mean().reset_index()

data1 = pd.concat([data_call_sms,data_else]).sort_values(by = [pacient_ID,time,attribute])



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
        values = list(data1.loc[data1['variable'] == 'appCat.entertainment']['value'])
        plt.figure()
        plt.hist(values)        
        plt.savefig('plots/hist_' + var + '.png')
        

plots(data1,variables)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


            
   
    
    
    