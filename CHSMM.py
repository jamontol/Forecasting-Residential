y# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:15:32 2019

@author: jamontol
"""

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve    
from sklearn.model_selection import learning_curve
from Plot_validation import plot_validation      
from Plot_learning import plot_learning
import numpy as np  

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

class CHSMM():
    
    
    def __init__(self, X_epoch, D_epoch, Z_epoch, minute_res, param, local_time_date):    
    
        self.X_epoch = X_epoch
        self.D_epoch = D_epoch
        self.Z_epoch = Z_epoch
        self.local_time_date = local_time_date
        self.n_states = param['n_states']
        self.MLR_weighted = param['MLR_weighted']
        self.MLR_state = param['MLR_state']
        self.state_cond = param['state_cond']
        self.multiclass = param['multiclass']
        self.weights = param['weights']
        self.state_cond = param['state_cond']
        self.solver = param['solver']
        self.samples = param['samples']
        self.a = param['w_factor']
        self.AS_mul_lr = []
        self.AD_mul_lr = []
        self.minute_res = minute_res
        self.type_day =  param['type_day']
        self.type_hour =  param['type_hour']
        
        #self.Entrenar()
        
    def Train(self,  opt_train_X = 1, opt_train_D = 0.05, validation = None):
    
        
        ###########Generate training matrices for MNLR  ###############################
        
        AS_set_in = []
        AS_set_out=[]
        AS_set_x_in = []
        AS_set_x_out = []
        AD_set_in = []
        AD_set_out= []        
        
        weight_full_X = None
        weight_full_D = None
        weight_X = None
        weight_D = None
        
        if (self.MLR_weighted):
        
            clase_weights = 'balanced'  
            #clase_weights = None
            if self.weights:
                weight = [1+(dk/self.a) for dk in self.D_epoch]
                weight_full = weight[:-1]
            #weight_full = None
            #    dic_weight = {key: value for (key, value) in enumerate(weight)}
        else:
        
            clase_weights = None
        
        if (self.MLR_state): # Para establecer modelos orientados a state y calcular 
        
            for x in range(self.n_states):
                
                index_epoch_X = [i for i,state in enumerate(self.X_epoch) if state == x] 
                
                # [y for x in Z_epoch for y in x]
                
                AS_set_x_in = [[self.X_epoch[ind], self.D_epoch[ind], self.Z_epoch[ind+1]] if self.state_cond else [self.X_epoch[ind], self.D_epoch[ind]] for ind in index_epoch_X[:-1]]
                AS_set_x_out = [self.X_epoch[ind+1] for ind in index_epoch_X[:-1]]
                
                index_epoch_D = [i for i,state in enumerate(self.X_epoch[1:]) if state == x]         
                
                AD_set_x_in = [[self.X_epoch[ind], self.D_epoch[ind], self.X_epoch[ind+1], self.Z_epoch[ind+1]]  if self.state_cond else [self.X_epoch[ind], self.D_epoch[ind], self.X_epoch[ind+1]] for ind in index_epoch_D[:-1]]
                AD_set_x_out = [self.D_epoch[ind+1] for ind in index_epoch_D[:-1]]    
                
                AS_set_in.append(AS_set_x_in) 
                AS_set_out.append(AS_set_x_out) 
            
                AD_set_in.append(AD_set_x_in) 
                AD_set_out.append(AD_set_x_out) 
                
                             
                # Entrenamiento óptimo
                
                if not(validation == None):
                    
                    if (validation == 'kf'):
                        CV = KFold(n_splits=3)
                    elif (validation == 'rkf'):
                        CV = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
                    elif (validation == 'loo'):
                        CV = LeaveOneOut()
                    elif (validation == 'lpo'):
                        CV = LeavePOut(p=2)
                    elif (validation == 'ss'):
                        CV = ShuffleSplit(n_splits=2, test_size=0.25, random_state=0)   
                    elif (validation == 'skf'):
                        CV = StratifiedKFold(n_splits=2)

                    
                    AS_mul_lr_x = LogisticRegressionCV(multi_class = self.multiclass, solver = self.solver,  cv = CV, max_iter = 10000)
                    AD_mul_lr_x = LogisticRegressionCV(multi_class = self.multiclass, solver = self.solver,  cv = CV, max_iter = 10000)

                    training_range = np.linspace(.1, 1.0, 5)
  
                    ## Learning curve
                    
                    if (False):
                    
                        #np.hstack
                        [train_sizes_AS, train_scores_AS, valid_scores_AS] = learning_curve(AS_mul_lr_x, np.array(AS_set_x_in), np.array(AS_set_x_out), train_sizes = training_range, cv=None, n_jobs=None)
                        [train_sizes_AD, train_scores_AD, valid_scores_AD] = learning_curve(AD_mul_lr_x, np.array(AD_set_x_in), np.array(AD_set_x_out), train_sizes = training_range, cv=None, n_jobs=None)
                        
                        opt_train_X = training_range[np.argmax(np.mean(valid_scores_AS,axis=1))] 
                        opt_train_D = training_range[np.argmax(np.mean(valid_scores_AD,axis=1))]
                
                else:
                
                    AS_mul_lr_x = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver, class_weight = clase_weights, C=10, max_iter = 10000)#.fit(np.array(AS_train_x_in), np.array(AS_train_x_out), sample_weight = weight_X)
                    AD_mul_lr_x = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver, class_weight = clase_weights, C=10, max_iter = 10000)#.fit(np.array(AD_train_x_in), np.array(AD_train_x_out), sample_weight = weight_D)
                                     

                AS_train_x_in = AS_set_x_in[:int(len(AS_set_x_in)*opt_train_X)]
                AD_train_x_in = AD_set_x_in[:int(len(AD_set_x_in)*opt_train_D)]
                AS_train_x_out = AS_set_x_out[:int(len(AS_set_x_out)*opt_train_X)]
                AD_train_x_out = AD_set_x_out[:int(len(AD_set_x_out)*opt_train_D)]
                
                index_epoch_X_train = index_epoch_X[:int(len(index_epoch_X[:-1])*opt_train_X)]
                index_epoch_D_train = index_epoch_D[:int(len(index_epoch_D[:-1])*opt_train_D)]
                
                if (self.MLR_weighted and self.weights):
                
                    weight_X = [weight[ind] for ind in index_epoch_X_train[:]]
                    weight_D = [weight[ind] for ind in index_epoch_D_train[:]]
                
                try:
                
                    AS_mul_lr_x = AS_mul_lr_x.fit(np.array(AS_train_x_in), np.array(AS_train_x_out), sample_weight = weight_X)
                    AD_mul_lr_x = AD_mul_lr_x.fit(np.array(AD_train_x_in), np.array(AD_train_x_out), sample_weight = weight_D)
                
                except:
                    
                    return 'Error'
                
                self.AS_mul_lr.append(AS_mul_lr_x)
                self.AD_mul_lr.append(AD_mul_lr_x)
        
        else:
                    
            # aS con (xk-1, dk-1, zk --> xk)
            for k, (dk,xk, zk1) in enumerate(zip(self.D_epoch[:-1], self.X_epoch[:-1], self.Z_epoch[1:])):
                
                if self.state_cond:
                    AS_set_in.append(np.hstack([xk, dk, zk1]))
                else:
                    AS_set_in.append([xk, dk])           
                
                AS_set_out.append(self.X_epoch[k+1])
            
            # aD con (xk-1, dk-1, xk, zk --> dk)
            for k, (dk,xk,xk1,zk1) in enumerate(zip(self.D_epoch[:-1], self.X_epoch[:-1], self.X_epoch[1:], self.Z_epoch[1:])):
                
                if self.state_cond:
                    AD_set_in.append(np.hstack([xk, dk, xk1, zk1]))
                else:
                    AD_set_in.append([xk, dk, xk1]) 
                    
                AD_set_out.append(self.D_epoch[k+1])
                
            # Train multinomial logistic regression model
                
            #self.AS_mul_lr = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver, class_weight = clase_weights)#.fit(np.array(AS_train_in), np.array(AS_train_out),sample_weight = weight_full)
            #self.AD_mul_lr = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver,  class_weight = clase_weights)#.fit(np.array(AD_train_in), np.array(AD_train_out),sample_weight = weight_full)
        
            ## Validation curve
            
            if not(validation == None):
                
                if (validation == 'kf'):
                    CV = KFold(n_splits=2)
                elif (validation == 'rkf'):
                    CV = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
                elif (validation == 'loo'):
                    CV = LeaveOneOut()
                elif (validation == 'lpo'):
                    CV = LeavePOut(p=2)
                elif (validation == 'ss'):
                    CV = ShuffleSplit(n_splits=2, test_size=0.25, random_state=0)   
                elif (validation == 'skf'):
                    CV = StratifiedKFold(n_splits=2)

                self.AS_mul_lr = LogisticRegressionCV(multi_class = self.multiclass, solver = self.solver,  cv = CV, max_iter = 10000)#, class_weight = clase_weights)#.fit(np.array(AS_train_x_in), np.array(AS_train_x_out), sample_weight = weight_X)
                self.AD_mul_lr = LogisticRegressionCV(multi_class = self.multiclass, solver = self.solver,  cv = CV, max_iter = 10000)#, class_weight = clase_weights)#.fit(np.array(AS_train_x_in), np.array(AS_train_x_out), sample_weight = weight_X)

                if (False):
                        
                    training_range = np.linspace(.1, 1.0, 5)
                    
                    [train_sizes_AS, train_scores_AS, valid_scores_AS] = learning_curve(self.AS_mul_lr, np.array(AS_set_in), np.array(AS_set_out), train_sizes = training_range, cv=5, n_jobs=4)
                    [train_sizes_AD, train_scores_AD, valid_scores_AD] = learning_curve(self.AD_mul_lr, np.array(AD_set_in), np.array(AD_set_x_out), train_sizes = training_range, cv=5, n_jobs=4)
                
                    opt_train_X = training_range[np.argmax(np.mean(valid_scores_AS,axis=1))] 
                    opt_train_D = training_range[np.argmax(np.mean(valid_scores_AD,axis=1))]
#           
            else:
                
                self.AS_mul_lr = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver, class_weight = clase_weights, max_iter = 10000)
                self.AD_mul_lr = LogisticRegression(random_state=0, multi_class = self.multiclass, solver = self.solver, class_weight = clase_weights, max_iter = 10000)
#
                    
            AS_train_in = AS_set_in[:int(len(AS_set_in)*opt_train_X)]
            AD_train_in = AD_set_in[:int(len(AD_set_in)*opt_train_D)]
            AS_train_out = AS_set_out[:int(len(AS_set_out)*opt_train_X)]
            AD_train_out = AD_set_out[:int(len(AD_set_out)*opt_train_D)]
            
        
            if (self.MLR_weighted and self.weights):
            
                weight_full_X = weight_full[:int(len(weight_full)*opt_train_X)]
                weight_full_D = weight_full[:int(len(weight_full)*opt_train_D)]
            
            try:
            
                self.AS_mul_lr = self.AS_mul_lr.fit(np.array(AS_train_in), np.array(AS_train_out),sample_weight = weight_full_X)
                                
                self.AD_mul_lr = self.AD_mul_lr.fit(np.array(AD_train_in), np.array(AD_train_out),sample_weight = weight_full_D)
    
            except:
                
                return 'Error'

        return 0


 
    def Validation(self, scoring = "accuracy", param = "C", param_range = np.logspace(-3, 2, 6), training_range = np.linspace(.1, 1.0, 5)):
        
        ## Validation curve      
        perc_train_X = []
        perc_train_D = []
    
        for x in range(len(self.AS_mul_lr)):
    
            [train_scores_AS, valid_scores_AS]  = validation_curve(self.AS_mul_lr[x], np.array(self.AS_set_in[x]), np.array(self.AS_set_out[x]), param_name = param, param_range = param_range, cv=5, scoring="accuracy", n_jobs=4)        
            plot_validation(train_scores_AS, valid_scores_AS, param, param_range, MLR = 'state'+str(x))
            [train_scores_AD, valid_scores_AD]  = validation_curve(self.AD_mul_lr[x], np.array(self.AD_set_in[x]), np.array(self.AD_set_out[x]), param_name = param, param_range = param_range, cv=5, scoring="accuracy", n_jobs=4)        
            plot_validation(train_scores_AD, valid_scores_AD, param, param_range, MLR = 'Duracion'+str(x))
    #        
            ## Learning curve
            
            [train_sizes_AS, train_scores_AS, valid_scores_AS] = learning_curve(self.AS_mul_lr[x], np.array(self.AS_set_in[x]), np.array(self.AS_set_out[x]), train_sizes = training_range, cv=5, n_jobs=4)
            plot_learning(train_scores_AS, valid_scores_AS, training_range, MLR = 'state'+str(x))
            [train_sizes_AD, train_scores_AD, valid_scores_AD] = learning_curve(self.AD_mul_lr[x], np.array(self.AD_set_in[x]), np.array(self.AD_set_out[x]), train_sizes = training_range, cv=5, n_jobs=4)
            plot_learning(train_scores_AD, valid_scores_AD, training_range, MLR = 'Duration'+str(x))

            opt_train_X = training_range[np.argmax(np.mean(valid_scores_AS,axis=1))] 
            opt_train_D = training_range[np.argmax(np.mean(valid_scores_AD,axis=1))]
            
            perc_train_X.append(opt_train_X)
            perc_train_D.append(opt_train_D)
           
        return  [np.mean(perc_train_X), np.mean(perc_train_D)]  
    

    def Predict(self, t, T, X_epoch, D_epoch, Emission):
            
        n = 0        
        suma=0
        k=0
        
    
        ind_cum = (np.cumsum(D_epoch) >= t)
        
        if any(ind_cum): 
            k = np.argmax(ind_cum)  #Retorna el primer valor mayor que t
        else:
            k = len(D_epoch)
        
        ts = sum(D_epoch[:k]) # t ini de epoch actual
        
        xk = X_epoch[k]
        xk_1 = X_epoch[k-1] 
        #xk = X_epoch[1]
        dk_1 = D_epoch[k-1]
        
        
        ############## Construir zk ################
        day_week = self.local_time_date[ts].weekday()
        if (day_week < 5): day = 0
        else: day = 1
        
        if (self.type_hour and self.type_day):
            zk = [np.floor(self.minute_res[ts]/self.samples), day]
        elif (self.type_hour):
            zk = np.floor(self.minute_res[ts]/self.samples)
        elif (self.type_day):
            zk = day
        ##########################################
    
        if (self.MLR_state == False):
        
            if (self.state_cond == False):
                
                dk = self.AD_mul_lr.predict(np.array([xk_1, dk_1, xk]).reshape(1, -1))[0]
            
            else:
                
                dk = self.AD_mul_lr.predict(np.hstack(np.array([xk_1, dk_1, xk, zk])).reshape(1, -1))[0]
        else:
            
            if (self.state_cond == False):
            
                dk = self.AD_mul_lr[xk].predict(np.array([xk_1, dk_1, xk]).reshape(1, -1))[0]
            
            else:
                              
                dk = self.AD_mul_lr[xk].predict(np.hstack(np.array([xk_1, dk_1, xk, zk])).reshape(1, -1))[0]
        
        if (dk < t - ts + 1): 
        
            dk = t - ts + 1
        
        Yt = []
        D = []
        
        for i in range(t, ts+dk): # Emissions in current state
                
                Yt.append(Emission[xk])
        
        xn_1 = xk
        dn_1 = dk
    
    
        tau = ts+dk 
        D.append(tau)
            
        while(tau <= t+T): #(ts+dk+T)):
        
        ############## Build zn ################
            day_week = self.local_time_date[tau].weekday()
            if (day_week < 5): day_n = 0
            else: day_n = 1
            
            if (self.type_hour and self.type_day):
                zn = [np.floor(self.minute_res[tau]/self.samples), day_n]
            elif (self.type_hour):
                zn = np.floor(self.minute_res[tau]/self.samples)
            elif (self.type_day):
                zn = day_n
        ##########################################
            
            
            if (self.MLR_state == False):
                
                if (self.state_cond == False):
                
                    xn = self.AS_mul_lr.predict(np.array([xn_1, dn_1]).reshape(1, -1))[0] 
                    dn = self.AD_mul_lr.predict(np.array([xn_1, dn_1, xn]).reshape(1, -1))[0]
                
                else:
                                        
                    xn = self.AS_mul_lr.predict(np.hstack(np.array([xn_1, dn_1, zn])).reshape(1, -1))[0] 
                    dn = self.AD_mul_lr.predict(np.hstack(np.array([xn_1, dn_1, xn, zn])).reshape(1, -1))[0]        
            else:
                
                if (self.state_cond == False):
                
                    xn = self.AS_mul_lr[xn_1].predict(np.array([xn_1, dn_1]).reshape(1, -1))[0] 
                    dn = self.AD_mul_lr[xn].predict(np.array([xn_1, dn_1, xn]).reshape(1, -1))[0]
                
                else:
                                        
                    xn = self.AS_mul_lr[xn_1].predict(np.hstack(np.array([xn_1, dn_1, zn])).reshape(1, -1))[0] 
                    dn = self.AD_mul_lr[xn].predict(np.hstack(np.array([xn_1, dn_1, xn, zn])).reshape(1, -1))[0]        
                       
            D.append(dn)
            
            #tau = min(t+np.sum(D),t+T)  #En el paper está como max (¿mal?)
            tau = np.sum(D)  #En el paper está como max (¿mal?)
            
            for i in range(tau-dn+1, tau+1,1):
                
                Yt.append(Emission[xn]) #param*array([1, ws])
        
            xn_1 = xn
            dn_1 = dn
            
        return Yt
     
        