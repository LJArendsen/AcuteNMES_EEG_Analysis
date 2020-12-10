#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:01:54 2020

@author: lauraarendsen
"""

import os
import mne

#%% Concatenate multiple files for a single participant (raw-files). 

"""
This code was used for the few exeptions were several recording files were present for a participant
after concatenating the raw files, the usual preproc and epoching pipeline can be applied
"""

# NOTE: Manually change the file names for each participant
raw_dir = "/mnt/data/Laura/Tuebingen/Inspiration_Pain_Project/AcuteNMES_Study/RawData/"
os.chdir(raw_dir)

filename_1 = 'ScSe_R001.fif'
filename_2 = 'ScSe_R002.fif'
#filename_3 = 'MiAn_R003.fif'
#filename_4 = 'MiAn_R004.fif'

raw_1 = mne.io.read_raw_fif(filename_1)
raw_2 = mne.io.read_raw_fif(filename_2)
#raw_3 = mne.io.read_raw_fif(filename_3)
#raw_4 = mne.io.read_raw_fif(filename_4)

raw_total = mne.concatenate_raws([raw_1, raw_2])
#raw_total = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4])

raw_total.save(raw_dir+'ScSe_R001_R002.fif', overwrite = False, split_size='2GB')