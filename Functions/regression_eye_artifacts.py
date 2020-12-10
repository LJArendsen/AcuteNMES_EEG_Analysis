#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:01:31 2020

@author: marius keute
"""

import mne
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS

def regress_out_pupils(raw, ocular_channels = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'], method = 'PCA'):
    
    """
    raw: Continuous raw data in MNE format
    ocular_channels: can be labels of EOG channels or EEG channels close to the
        eyes if no EOG was recorded
    method: how to combine the ocular channels. Can be 'PCA', 'mean', or 'median'.
    """
    
    raw_data = raw.get_data(picks = 'eeg')
    ocular_data = raw.get_data(picks = ocular_channels)
    
    if method == 'PCA':
        pca = PCA()
        comps = pca.fit_transform(ocular_data.T)
        ocular_chan = comps[:,0]
    elif method == 'mean':
        ocular_chan = np.mean(ocular_data, axis = 0)
    elif method == 'median':
        ocular_chan = np.median(ocular_data, axis = 0)
    
    for ch in range(raw_data.shape[0]):
        m = OLS(raw_data[ch,:], ocular_chan)
        raw_data[ch,:] -= m.fit().predict()
    raw._data[:raw_data.shape[0],:] = raw_data
    return raw
