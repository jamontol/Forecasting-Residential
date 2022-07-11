# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:52:15 2019

@author: jamontol
"""

import numpy as np

def mae_metric(actual, predicted):
    
    sum_error = 0.0
    for i in range(len(actual)):

        sum_error += abs(predicted[i] - actual[i])

    mae = sum_error/float(len(actual))
    print(mae)
    return mae[0]
      
        
# Calculate root mean squared error
def rmse_metric(actual, predicted):
    
    sum_error = 0.0
    for i in range(len(actual)):
  
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    
    mean_error = sum_error / float(len(actual))
	
    rmse = np.sqrt(mean_error)
    print(rmse)
    return rmse[0]


def nrmse_metric(actual, predicted):
    
    sum_error = 0.0
	
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    rmse = np.sqrt(mean_error)
    
    nrmse = rmse/np.std(actual)
    print(nrmse)
    return nrmse[0]
    