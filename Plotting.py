# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:27:26 2018

@author: jamontol
"""
__author__ = 'Javier Monreal'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(Consumption, Prediction, t_ini, Ts, title):

    
    fig = plt.figure(figsize=(20,20))
    titulo = 'Consumption'+' '+title
    plt.title(titulo)    
    plt.grid()
    rango = range(Ts*t_ini,Ts*(len(Prediccion))+Ts*t_ini)
    rango = range(Ts*t_ini,Ts*(len(Prediccion))+Ts*t_ini)
    Consumo = np.repeat(Consumo,Ts)
    Prediccion = np.repeat(Prediccion,Ts)
    plt.plot(rango, Consumo ,'b', linewidth = 3 )
    plt.plot(rango, Prediccion,'r')
    xlabel = "Time [min]"
    plt.xlabel(xlabel)
    plt.ylabel("Consumption [kW]",color='k')    
    
    plt.show()

def plot_learning(train_scores, test_scores, train_sizes=np.linspace(.1, 1.0, 5), MLR=[]):

    plt.figure()
    
    titulo = "Learning Curve with MLR:" + MLR
    
    plt.title(titulo)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
        
    plt.grid()

    plt.ylim(0.0, 1.1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label="Training score", marker = 'o', color = "r",)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color = "r")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", marker = 'o', color = "g",)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color = "g")
    plt.legend(loc="best")
    
    plt.show()

def plot_validation(train_scores, test_scores, param, param_range, MLR=[]):

    plt.figure()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    titulo = "Validation Curve with MLR: " + MLR
    
    plt.title(titulo)
    plt.xlabel(str(param))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    #plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    #plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    
    plt.show()