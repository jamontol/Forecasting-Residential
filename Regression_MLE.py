# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:42:13 2019

@author: jamontol
"""

from scipy.optimize import minimize
import numpy as np


# Define the likelihood function where params is a list of initial parameter estimates
def regressLL(params):

    import scipy.stats as stats
    # Resave the initial parameter guesses
    b0 = params[0]
    b1 = params[1]
    sd = params[2]

    # Calculate the predicted values from the initial parameter guesses
    yPred = b0 + b1*X

    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd
    logLik = -np.sum( stats.norm.logpdf(Y, loc=yPred, scale=sd) )

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)

# Make a list of initial parameter guesses (b0, b1, sd)    


def Regression(y, x, initParams, method='L-BFGS-B'):

    global X, Y
    #initParams = [1, 1, 1]    
    # Run the minimizer
    results = minimize(regressLL, initParams, method='nelder-mead')
    
    print(results.x)
    
    return results
    
