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

################################## ATTRIBUTE AGGREGATION ####################################
mean_var = set(['mood','activity','circumplex.arousal','circumplex.valence'])
aggregated_pacients, labels = dprep.aggregate_and_get_labels(pacients_original, ids, mean_var)


################################## SPLIT DATA TRAIN/TEST ####################################
X_train, Y_train, X_test, Y_test = tt.train_test(aggregated_pacients, labels, ids, 0.65)


############################### STANDARDIZATION and PCA ######################################
X_train_st, X_test_st  = dprep.standardization(X_train, X_test, ids)
X_train_pca, X_test_pca  = dprep.pca(X_train_st, X_test_st, ids)


##############################  Tests ###########################
bench_predictions = tt.benchmark(pacients_original, Y_test, ids)
svr_predictions = tt.svm_regression(X_train_pca, Y_train, X_test_pca,                          
                                    ids,   
                                    kernel = 'rbf',
                                    degree = 2)

bench = tt.prediction_stats(bench_predictions, Y_test, ids)
svr = tt.prediction_stats(svr_predictions, Y_test, ids)


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









pid = 'AS14.01'
aux = mse(bench_predictions[pid], Y_test[pid])











