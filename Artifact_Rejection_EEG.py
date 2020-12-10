#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:33:21 2020

@author: lauraarendsen
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from AcuteNMES_EEG_Analysis.Code.Functions import visual_inspection as VIF

#%% Artifact correction step 1 : interpolation of bad channels

epochs_dir = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Epochs/"
os.chdir(epochs_dir)

dictionary_bad_channels = {'EuSa': ['T7'], 'ZaHa': ['AF7', 'AF8'], 'HeDo': ['TP9', 'TP10'], 
                           'LiLu': ['T7', 'T8'], 'MeAm': ['AF7', 'AF8'], 'NeMa': ['FT9', 'FT10'], 
                           'ScSe': ['T7', 'T8', 'TP9', 'TP10'], 'VeVa': ['AF7', 'T7', 'T8']}

# load epoch_preproc data files
files_epochs = ['epochs_preproc_VeVa_R001.fif', 'epochs_preproc_RaLu_R001.fif', 'epochs_preproc_NeMa_R001.fif', 'epochs_preproc_EuSa_R001.fif', 
                'epochs_preproc_ZhJi_R002.fif', 'epochs_preproc_KoAl_R008_R009_R010.fif', 'epochs_preproc_HeDo_R010_R011.fif', 
                'epochs_preproc_ZiAm_R001.fif', 'epochs_preproc_LiLu_R001.fif', 'epochs_preproc_BrTi_R001.fif', 'epochs_preproc_StLa_R001.fif', 
                'epochs_preproc_MiAn_R001_R002_R003_R004.fif', 'epochs_preproc_RuMa_R001.fif', 'epochs_preproc_MeAm_R001_R002.fif', 
                'epochs_preproc_LaCa_R001.fif', 'epochs_preproc_BaJo_R001.fif', 'epochs_preproc_ScSe_R001_R002.fif', 'epochs_preproc_UtLi_R001.fif', 
                'epochs_preproc_HaSa_R001_R002.fif', 'epochs_preproc_WoIr_R001.fif', 'epochs_preproc_ZaHa_R001.fif']

save_folder = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Artifact/"
               
# create subject_ID based on filename and check if it's present in the BadChannels dictionary 
for f in files_epochs:
    subject_ID = f[15:19]
    print(f[15:19])
    if subject_ID in dictionary_bad_channels:
        print(f'{subject_ID} is present in the list')
        # load file
        epochs = mne.read_epochs(f, preload=True)
        # create a 'bads' list for this participant (f)
        epochs.info['bads'] = dictionary_bad_channels[subject_ID]
        print(epochs.info['bads'])
        # Get montage (layout)
        epochs.set_montage('easycap-M1')
        # interpolate 'bads'
        epochs_art1 = epochs.interpolate_bads(reset_bads=True, mode='accurate')
        epochs_art1.save(save_folder+'art1_'+f, overwrite = False, split_size='2GB')       
    else:
        print(f'{subject_ID} is not present in the list')
        # no changes, just save unchanged, using the same variable name as for interpolated files, to be used for art.rej. 3 step
        epochs = mne.read_epochs(f, preload=True)
        epochs_art1 = epochs.copy()
        epochs_art1.save(save_folder+'art1_'+f, overwrite = False, split_size='2GB')

#%% Artifact rejection step 2 : use a threshold to reject epochs with remaining artifacts

"""
identify epochs to be removed, based on a set threshold (automatic detection). Currently, I am using amplitude range as 
my measure and >60 microvolt as the threshold to reject. For now set up to just run a single file at the time --> make sure 
to manually adjust filename to load and filename to save
"""


filename = '/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Artifact/art1_epochs_preproc_KoAl_R008_R009_R010.fif'

epochs = mne.read_epochs(filename, preload=True)

# epoch_SD = np.std(np.mean(eeg_epochs.get_data(), axis=1), axis=1) #calculate one summary score, e.g., variance, for each epoch
# epoch_var = np.var(np.mean(eeg_epochs.get_data(), axis=1), axis=1) 
epoch_range = np.ptp(np.mean(epochs.get_data(), axis=1), axis=1)

tbexcluded = VIF.visual_inspection(epoch_range, indexmode = 'exclude') 
#this allows you to visually select epochs that you want to exclude, and returns their indices.
#if you want to get the indices of the non-excluded epochs instead, set indexmode = 'keep'

# remove the identified 'bad' epochs
epochs_art2 = epochs.drop(tbexcluded)

save_folder = '/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/AcuteNMES_EEG_Analysis/Artifact/'

epochs_art2.save(save_folder+'clean_KoAl_R008_R009_R010.fif', overwrite = False, split_size='2GB')


#%% Finally, remaining steps to carry out ...?

# apply re-refencing? I would usually apply an average reference (freq. analysis), but not sure if this is best for ERP analysis.
# However, a linked-mastoid would be tricky as we did not apply separate mastoid electrodes...








