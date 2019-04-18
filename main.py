import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_preprocessing as dprep
import train_test as tt

DATA_PATH = 'data/dataset_mood_smartphone.csv'

#################################### LOAD DATA #########################################
data, ids, variables = dprep.load_data(DATA_PATH)

############################ GET STATISTICS AND CLEAN DATA ###############################
data = dprep.remove_5(data, variables)
table = dprep.stats(data, variables)
table2 = dprep.stats(data, variables) 


########################## GROUP DATA BY PACIENT, DAY, VARIABLE ###############################    
data_call_sms = data.loc[~data['variable'].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby(['id','time','variable']).sum().reset_index()
data_else = data.loc[data['variable'].isin(['mood','activity','circumplex.arousal','circumplex.valence'])].groupby(['id','time','variable']).mean().reset_index()
data = pd.concat([data_call_sms,data_else]).sort_values(by = ['id','time','variable'])

################################ DEALING WITH MISSING DATA ################################### 
data =  dprep.fill_missing_data(data, ids, variables)
data.to_csv(path_or_buf = 'data/data.csv')


###################### REMOVE DAYS WITH NO RELEVANT DATA(SPARSE DATA) #########################
data = pd.crosstab(index = [data['id'],data['time']], 
                   columns = data['variable'], values = data['value'], 
                   aggfunc = lambda x:x).reset_index()

# Swap 3rd column with mood, so later mood is the first column
columns_new = data.columns.values.copy() 
columns_new[2], columns_new[18] = columns_new[18], columns_new[2]
data = data[columns_new]

data = dprep.remove_days(data,ids)
data.to_csv(path_or_buf = 'data/data.csv')


################################## CORRELATION MATRXIX ####################################
total_corr, pacient_corr, pacient_mood_corr = dprep.correlation(data,ids)

####################### INDIVIDUALIZE DATA AND REMOVE INDIVIDUAL NaN DIMS ######################
pacients_original = dprep.individualize(data, ids, pacient_mood_corr)


#################################### STANDARDIZATION #########################################
mean_var = set(['activity','circumplex.arousal','circumplex.valence', 'time','id'])
pacients_standardized = dprep.standardization(pacients_original, ids, mean_var)



######################################## PCA ##############################################
pacients_pca = dprep.pca(pacients_standardized,ids)



##############################  Tests ###########################
    
X_train, Y_train, X_test, Y_test = tt.train_test_pca(pacients_pca,
                                                     pacients_original, 
                                                     ids, 
                                                     0.6)

_, _, X_test_orig, _ = tt.train_test_individual_pacient(pacients_original,
                                         pacients_original, 
                                         ids, 
                                         0.6)


X_train_all, Y_train_all, X_test_all, Y_test_all = tt.train_test_all(X_train, Y_train, X_test, Y_test, ids)


bench_predictions = tt.benchmark(X_test_orig,ids, period = 3)
svr_predictions = tt.svm_regression(X_train, Y_train, X_test,
                                    X_train_all, Y_train_all, X_test_all, 
                                    ids,            
                                    degree = 2)

bench, bench_all = tt.prediction_stats(bench_predictions, Y_test, Y_test_all, ids)
svr,svr_all = tt.prediction_stats(svr_predictions, Y_test,Y_test_all,ids)


plt.figure()
plt.plot(bench, label = 'bench')
plt.plot(svr, label = 'svr')
plt.title('MSE for each patient')
plt.xlabel("Patient number")
plt.ylabel("MSE")
plt.legend(loc='upper left')
plt.show()


def plots(data1,variables):    
    for var in variables:
        values = list(data1.loc[data1['variable'] == var]['value'])
        plt.figure()
        plt.hist(values)        
        plt.savefig('plots/hist_' + var + '.png')




















