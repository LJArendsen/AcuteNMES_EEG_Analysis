#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:00:37 2020

@author: lauraarendsen
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from AcuteNMES_EEG_Analysis.Code.Functions import regression_eye_artifacts as REA


#%% Preproc steps, continuous raw data

""" 
select EEG channels, filtering, and regress out eye artifacts

step 1: raw continous data is loaded and the EEG channels are selected and saved
(channel 1-64 out of 73 channels in total. Chan 65-73 are EMG + ECG)

step 2: filters are applied on the continuous eeg channels

step 3: Linear regression to remove ocular artifacts from EEG data, using:
    def regress_out_pupils(raw, ocular_channels = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'], method = 'PCA'): ...

"""

raw_dir = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/RawData/"

files = ['ZiAm_R001.fif', 'RaLu_R001.fif', 'KoAl_R008_R009_R010.fif', 'LiLu_R001.fif', 
         'ZhJi_R002.fif', 'EuSa_R001.fif', 'HeDo_R010_R011.fif', 'ScSe_R001_R002.fif',
         'BaJo_R001.fif', 'NeMa_R001.fif', 'MiAn_R001_R002_R003_R004.fif', 'ZaHa_R001.fif',
         'BrTi_R001.fif', 'UtLi_R001.fif', 'RuMa_R001.fif', 'MeAm_R001_R002.fif',
         'WoIr_R001.fif', 'VeVa_R001.fif', 'StLa_R001.fif', 'HaSa_R001_R002.fif', 'LaCa_R001.fif']

for f in files:
    # load raw fif file
    print('loading '+str(f))
    raw = mne.io.read_raw_fif(raw_dir+f)
    raw.load_data()
    
    # step 1: select EEG channels
    raw_eeg = raw.copy()
    print('Number of channels in raw_eeg:')
    print(len(raw_eeg.ch_names), end=' → drop nine → ')
    raw_eeg = raw_eeg.drop_channels(['EDC_L', 'EDC_R', 'ECR_L', 'ECR_R', 'FCR_L', 'FCR_R', 'FDS_L', 'FDS_R', 'EKG'])
    print(len(raw_eeg.ch_names))
    
    # step 2: apply filters
    sFreq = raw_eeg.info['sfreq']
    raw_eeg_notch = raw_eeg.notch_filter((50,100,150))    # notch filter, line noise
    raw_eeg_notch_hp = raw_eeg_notch.filter(l_freq=0.5, h_freq=None)    # high-pass filter at 0.5 Hz (using default filter settings = fir)  
    raw_eeg_filtered = raw_eeg_notch_hp.filter(l_freq=None, h_freq=40)    # low-pass filter at 40 Hz.
    
    # step 3: apply regression for ocular artifacts
    raw_regr = REA.regress_out_pupils(raw_eeg_filtered)
    # raw_regr.plot(n_channels=64)
  
    # save this preproc data file with filtered and regressed continuous data to the AcuteNMES_EEG_Analysis/Preproc folder
    save_folder = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Preproc/"
    raw_eeg_filtered.save(save_folder+'preproc_'+f, overwrite = False, split_size='2GB')
    


#%% Create epochs

""" 
create epochs, reject remaining artifacts using semi-auto approach

step 1: events + event_ids were created from the annotations present in the raw data files. 
step 2: These events were used to create epochs from -0.5 to +1s around each event. 

"""
    
preproc_dir = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Preproc/"

files_preproc = ['preproc_ZiAm_R001.fif', 'preproc_RaLu_R001.fif', 'preproc_KoAl_R008_R009_R010.fif', 'preproc_LiLu_R001.fif', 
         'preproc_ZhJi_R002.fif', 'preproc_EuSa_R001.fif', 'preproc_HeDo_R010_R011.fif', 'preproc_ScSe_R001_R002.fif',
         'preproc_BaJo_R001.fif', 'preproc_NeMa_R001.fif', 'preproc_MiAn_R001_R002_R003_R004.fif', 'preproc_ZaHa_R001.fif',
         'preproc_BrTi_R001.fif', 'preproc_UtLi_R001.fif', 'preproc_RuMa_R001.fif', 'preproc_MeAm_R001_R002.fif',
         'preproc_WoIr_R001.fif', 'preproc_VeVa_R001.fif', 'preproc_StLa_R001.fif', 'preproc_HaSa_R001_R002.fif', 'preproc_LaCa_R001.fif']
    
for f in files_preproc:    
    print('loading '+str(f))   
    raw_preproc = mne.io.read_raw_fif(preproc_dir+f, preload=True)
    raw_preproc.load_data()
    
    # Step 1: create events from the annotations present in the raw file
    (events_from_annot, event_dict) = mne.events_from_annotations(raw_preproc)

    # Step 2: create epochs based on the events, from -500 to 1000 ms
    epochs = mne.Epochs(raw_preproc, events_from_annot, event_id=event_dict,
                        tmin=-0.5, tmax=1, reject=None, preload=True,
                        baseline=None, detrend=None)

    # save the preproc-epoched data file to the AcuteNMES_EEG_Analysis/Epochs folder
    save_folder = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Epochs/"
    epochs.save(save_folder+'epochs_'+f, overwrite = False, split_size='2GB')
    
    
#%% Additional step, for participant ZaHa only

"""
phase determination eeg cables were not attached for the first ~150 trials for ZaHa, therefore (to be conservative),
the first 155 trials were removed for this participant
"""

epochs_dir = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Epochs/"

filename = 'epochs_preproc_ZaHa_R001.fif'
epochs = mne.read_epochs(epochs_dir+filename, preload=True)

drop_indices = np.arange(155)
epochs = epochs.drop(drop_indices)

# save this preproc data file (overwrite the previous epochs_preproc_ZaHa file that still included the first 155 trials)
save_folder = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Epochs/"
epochs.save(save_folder+filename, overwrite = True, split_size='2GB')
    
    
    