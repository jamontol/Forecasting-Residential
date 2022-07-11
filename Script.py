# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:29:36 2019

@author: jamontol
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pymysql
from struct import *
import binascii
import Metrics as error
from sklearn.externals import joblib

import pickle

BBDD = 'PECAN' 

facility = 1
appliance = 'others'

emission_cond = False

day_type = True
hour_type = False#True
n_states = 3
n_data = 11550 #None
Ts = 2
ind_ini = int(60/Ts)
hour_samples = int(60/Ts)


db = pymysql.connect(host="",   # your host, usually localhost
                     user="",      # your username
                     port= ,
                     passwd="",    # your password
                     db="")       # name of the data base

Appliances = ['Oven', 'Fridge', 'Washing machine', 'microwave', 'Electric heating element']

try:
    cur = db.cursor()
except (AttributeError, pymysql.MySQLError):
    cur = db.cursor()


cur.execute("SELECT idAppliance FROM Appliance WHERE id_facility = %s AND tag = %s" , (facility, appliance)) 
idAppliance = cur.fetchone()[0]
cur.execute("SELECT * FROM valores_diarios WHERE Appliance = %s", idAppliance) 


raw_data = [(row[3], row[4]) for row in cur.fetchall()]
data = []     
local_time = []       

for row in raw_data:

    it = row.index
    hora = row[0]
    data_hex = binascii.hexlify(row[1]).decode('ascii')

    for k in np.arange(0,240,8): # 30 data/hour (Ts = 2 min)  
    
        dato_float = unpack('<f', bytes.fromhex(data_hex[k:k+8]))[0]  
        data.append(dato_float)
        
    local_time.extend([hora.replace(minute=Ts*k) for k in range(0,hour_samples)])
        
consumption = pd.Series(data[ind_ini:n_data]).values 
data =  pd.Series(data[ind_ini:n_data]) #pd.Series(data[:n_train])
local_time_date = local_time[ind_ini:]
data_consumption = data.values.reshape(-1,1) 

minuto_res = np.array([int(time.hour*hour_samples + time.minute/Ts)  for time in local_time_date])
        
#%% K-Means


from sklearn.cluster import KMeans
from itertools import count, groupby  
   
#TODO Estimate clusters number

f1=plt.figure()
plt.hist(x=data, bins=100, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Valor')
plt.ylabel('Frequencia')
plt.title('Histograma'+' '+'['+ appliance+ ']')
plt.show()

f2=plt.figure()
data.plot.hist(grid=True, bins=100, rwidth=0.9, color='#607c8e')

f3=plt.figure()
plt.plot(consumption)


kmeans = KMeans(n_clusters=n_states, random_state=0).fit(data_consumption[:1000])
#sec_states = kmeans.labels_

centroids = kmeans.cluster_centers_

sec_states = kmeans.predict(data_consumption)

############ States and durations ##########################

D_epoch = []
X_epoch = []
Z_epoch = []
epochs=()

index = 0

for k, g in groupby(enumerate(sec_states), lambda args: args[1]): #Detect intervals with same values
                            
    inter =  len(list(g)) 
    D_epoch.append(inter)
    X_epoch.append(k) 

    dia_semana = local_time_date[index].weekday()
    if (dia_semana < 5): dia = 0
    else: dia = 1
    
    if (hour_type and day_type):
        Z_epoch.append([np.floor(minuto_res[index]/hour_samples), dia])
    elif (hour_type):
        Z_epoch.append(np.floor(minuto_res[index]/hour_samples)) 
    elif (day_type):
        Z_epoch.append(dia)

    index = index + inter  

epochs =(X_epoch, D_epoch)    

index_epoch = []
index_states = []
data_state = []
average_state = []


# Number of epochs 

sum=0
epochs_train=0


#%% Gaussian emissions
   

from Regression_MLE import Regression

Y= []
X= []

Emission = []
for s in range(n_states):
    
    Y = [data_consumption[k] for k, state in enumerate(sec_states) if state ==s] ;
     
    ind_state = [i for i,state in enumerate(sec_states) if state == s] 
    
    Y = data_consumption[ind_state]
    
    if emission_cond:
        X = [[centroids[s],w] for w in W_emission[s]] # average and exogenous variable
        Emission.append(Regression(Y,X,[centroids[s],1,1])) 
        
    else:
        X = centroids[s]
        Emission.append(X)
    
#%% CHSMM

from Plotting import plot_data
import CHSMM as CHSMM

weights = True
emission_cond = False
MLR_solver = 'newton-cg'#'lbfgs'
MLR_multiclass = 'multinomial'#'ovr'
a=1 #Factor de podneracion

#########################

L_Retrain = [False, True]
L_cond_state = [False, True]
L_MLR_state = [False, True]
L_MLR_weighted = [False, True]
L_weight = [False, True]

METRICS = pd.DataFrame({'Conf':[], 'MAE':[], 'RMSE':[], 'NRMSE':[]})

D_train = 7
D_pred = 7

H = int(24*60/Ts) #update data

perc_train_X = 1
perc_train_D = 1

k=0
for d_train in range(7,8):

    t_ini = int((d_train*1440)/Ts) #Firs hour of next day
    Tsim = int(((D_pred+D_train)*1440)/Ts)

    for retrain in L_Retrain[:]:
    
        for cond_state in L_cond_state[:]:
    
            for MLR_state in L_MLR_state[:1]:
    
                for MLR_weighted in L_MLR_weighted[:]:
                    
                    for weights in L_weights:
                    
                        param = dict({'n_states': n_states, 'state_cond': state_cond, 'emission_cond': emission_cond, 'MLR_state': MLR_state, 'MLR_weighted': MLR_weighted, 'solver': MLR_solver, 'multiclass': MLR_multiclass, 'muestras': hour_samples, 'weights':weights, 'w_factor': a, 'day_type':day_type, 'hour_type':hour_type })
                        
                        title = ''
                        title += 'R='+ str(retrain)+','
                        title += 'C='+ str(cond_state)+','
                        title += 'E='+ str(MLR_state)+','
                        title += 'P='+ str(MLR_weighted)
                        
                        CONSUMPTION_PRED = []
                        
                        sum=0
                        epochs_train_ini=0
                        train = True
                        dia_ini = 0
                        
                        for t in range(t_ini,Tsim, H):
                            
                            z = np.floor(minuto_res[t]/hour_samples)
                                        
                            epochs_fin = 0
                            
                            # Determine currnet epoch 
                                
                            epochs_fin = np.argmax(np.cumsum(D_epoch) >= t) #Returs firt value bigger than t
                                
                            X_epoch_t = X_epoch[:epochs_fin+1] 
                            D_epoch_t = D_epoch[:epochs_fin]
                            Z_epoch_t = Z_epoch[:epochs_fin+1]
                            
                            if (z == 0 and train): #Re-train new day
                                
                                t_ini_train = int((dia_ini*1440)/Ts)
                                            
                                epochs_train_ini = np.argmax(np.cumsum(D_epoch) >= t_ini_train) #Returs firt value bigger than t
                                
                                Chsmm = CHSMM.CHSMM(X_epoch_t[epochs_train_ini:], D_epoch_t[epochs_train_ini:], Z_epoch_t[epochs_train_ini:], minuto_res, param, local_time_date) 
                                Chsmm.Train(opt_train_X = perc_train_X, opt_train_D = perc_train_D, validacion = None)
                                
                                dia_ini = dia_ini+1
                            
                            if (retrain): 
                                
                                train = True
                                
                            else:
                                
                                train = False
                        
                            consumption_pred = Chsmm.Predict(t, H, X_epoch_t, D_epoch_t, emission)
                                
                            CONSUMPTION_PRED.extend(consumption_pred[0:H])
                            
                        plot_data(consumption[t_ini:len(CONSUMPTION_PRED)+t_ini], CONSUMPTION_PRED, t_ini, Ts, title)
                        
                        MAE = error.mae_metric(consumption[t_ini:len(CONSUMPTION_PRED)+t_ini],CONSUMPTION_PRED)
                        RMSE = error.rmse_metric(consumption[t_ini:len(CONSUMPTION_PRED)+t_ini],CONSUMPTION_PRED)
                        NRMSE = error.nrmse_metric(consumption[t_ini:len(CONSUMPTION_PRED)+t_ini],CONSUMPTION_PRED)
                        
                        METRICAS.loc[k]=[title, MAE, RMSE, NRMSE]
                        
                        k=k+1

Metrics_file = str(facility)+str(appliance)+str(H)                  
with open(Metrics_file, 'wb') as f:
    pickle.dump(METRICS, f)


#%%
#### Save model ####

filename = BBDD +'_'+ str(facility) +'_'+ appliance +'_'+str(perc_train_X)+'-'+str(perc_train_D)   

joblib.dump(Chsmm, filename)
 



