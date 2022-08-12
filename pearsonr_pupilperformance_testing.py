# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:55:42 2022

@author: Anne
"""
import numpy as np
import scipy.stats

def ridge_error(x,y, r=0):
    ''' ridge_error(x,y, lambda):
    predicts y [N by m] from x [N by p] by ridge regression. 
    
    Returns:
    * fit error as a frac of variance (scalar), 
    * weight [p by m], 
    * prediction [N by m].'''
    
    # append to bottom of matrices to make ridge regression
    n = x.shape[0]
    p = 1
    m = 1
    x0 = np.block([[x, np.ones([n])]])
    x1 = np.block([[x, np.ones([n])], [r*np.eye(p), np.zeros([p, 1])]])
    y1 = np.block([[y],[np.zeros((p,m))]])
    w,_,_,_ = np.linalg.lstsq(x1,y1, rcond=None)
    yhat = x0@w
    return np.mean((y-yhat)**2)/np.var(y), w, yhat

JC047_performance = np.genfromtxt(r'C:\Users\Anne\data\JC047\20211001_114337\saved_data\JC047_performance_by_session.csv', delimiter=',')
JC047_pupil = np.genfromtxt(r'C:\Users\Anne\data\JC047\20211001_114337\saved_data\JC047_pupil_by_session.csv', delimiter=',')


S = 4
cc_storage = np.empty(S,dtype=object)
pval_storage = np.empty(S,dtype=object)

x = JC047_pupil
y = JC047_performance

for s in range(S): 
    # x[s] = JC047_pupil[s]
    # y[s] = JC047_performance[s]
    cc, pval = scipy.stats.pearsonr(x[s], y[s])
    cc_storage[s] = cc
    pval_storage[s] = pval
    print(cc, pval)