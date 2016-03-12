# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:33:35 2015

@author: Yujie

Some self-defined stat methods

"""

import numpy as np

def cal_R2(yobs, yhat, n=None, p=None): 
    """
    Calculate r2 and adjusted R2
    Return: a 1-D list with R2 and adjR2 (only when n and p are provided)
    """
    ss_tot = np.nansum((yobs - np.nanmean(yobs))**2.)
    ss_reg = np.nansum((yhat - np.nanmean(yobs))**2.)
    ss_res = np.nansum((yobs - yhat)**2.)
    r2 = 1. - ss_res/ss_tot
    if n is not None and p is not None:
        r2adj = r2 - (1 - r2)*(p*1./(n-p-1))
        return r2, r2adj
    return r2
    
def cal_RMSE(y, yhat):
    """
    Calculate rmse of modeled y
    """
    notNaNs = ~np.isnan(y)
    try:
        nsmp = y[notNaNs].shape[0]
    except:
        nsmp = len(y[notNaNs])
    return np.sqrt(np.nansum((y - yhat)**2.)/nsmp)

def cal_pctERR(y, yhat):
    ''' 
    Calculate percent error, weighted by y's values
    '''
    notNaNs = ~np.isnan(y)
    try:
        nsmp = y[notNaNs].shape[0]
    except:
        nsmp = len(y[notNaNs])
    tmp = np.abs((y[notNaNs] - yhat[notNaNs])/y[notNaNs])
    return np.sum(y*tmp)/(np.sum(y)*nsmp)
    
def rebin_mean(a, shape):
    ''' 
    Calculate the average of smaller matrix block when aggregating from high resolution
    data to low resolution.
    parameters:
        a     : 2-D ndarray of gridded data
        shape : the target resolution (e.g., (3600, 7200) -> (360, 720), the factor
                has to be an integer) 
    '''
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    
def rebin_majvote(a, shape):
    ''' 
    Calculate the majority of smaller matrix block (categorical variable) when 
    aggregating from high resolution data to low resolution.
    parameters:
        a     : 2-D ndarray of gridded data
        shape : the target resolution (e.g., (3600, 7200) -> (360, 720), the factor
                has to be an integer, and is equal for row and col) 
    '''
    from scipy.stats import mode
    fac = a.shape[0]//shape[0]
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            dum = a[i*fac:(i+1)*fac,j*fac:(j+1)*fac]
            most_frequent = mode(dum, axis=None)[0][0]
            out[i,j] = most_frequent
    return out

def cal_earthgridarea(resolution):
    '''
    Calculate cell area for each grid. 
    Parameters: 
        resolutiion : in degree (e.g., 0.5)
    return:
        areaa       : areaa of each grid in m2, norther-south direction
    '''    
    nrow = int(180./resolution); ncol = int(360./resolution)
    lat = np.arange(90,-90,-resolution)
    areaa = np.zeros((nrow/2, 1))
    rad = 6.371e6 # earth average radius is 6371km, unit m
    height = rad*2.*np.pi/ncol
    for i in range(1,nrow/2+1):
        hori = rad*np.cos(lat[i]*np.pi/180.)*2*np.pi/ncol
        areaa[i-1,0] = hori*height
    areaa = np.tile(areaa,(1,ncol))
    areaa = np.r_[areaa, np.flipud(areaa)]
    return areaa