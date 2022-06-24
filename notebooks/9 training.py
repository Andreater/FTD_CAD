#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:40:08 2022

@author: lab401
"""

"""
Here we perform multiple MONAI training.
"""

import sys
sys.path.append("C:/Users/LAB401/Analyses/NIFD_exp/r/")

from monai402 import monai_training, build_data_lists
from monai.transforms import AddChannel, Compose, ScaleIntensity, EnsureType, Resize #, SqueezeDim, RandRotate90, Resize

import pandas as pd

resize_ss = 150

# define transforms
train_transforms = Compose([
                            # SqueezeDim(dim = 2),
                            ScaleIntensity(),
                            AddChannel(),
                            #Resize((96, 96, 96)),
                            Resize(spatial_size = resize_ss,
                                   size_mode = "longest"),
                            # RandRotate90(),
                            EnsureType()
                            ])

val_transforms = Compose([
                          # SqueezeDim(dim = 2),
                          ScaleIntensity(),
                          AddChannel(), 
                          Resize(spatial_size = resize_ss,
                                 size_mode = "longest"),
                          EnsureType()
                          ])
#%%

models_dir = 'C:/Users/LAB401/Analyses/NIFD_exp/data/models/'
 
# this is metadata_full
metadata_df = pd.read_excel(r'C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/metadata_full.xlsx')

# first train
first_train_set_df = pd.read_excel(r'C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/first_training_set.xlsx')

# to store val_preds
predictions_dir = 'C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/MONAI predictions/'


#%%
train_index = build_data_lists(metadata            = metadata_df, 
                               experimental_groups = first_train_set_df, # this for first train
                               case_label          = "case",
                               needed_set          = 'first train') # must be 'first train' or 'train'

# rinomino le colonne
experimental_groups_df = metadata_df[['aug_id', 'experimental_group']]
experimental_groups_df.columns = ['id_aug', 'experimental_group']

val_index = build_data_lists(metadata              = metadata_df, 
                             experimental_groups   = experimental_groups_df, 
                             case_label            = "case",
                             needed_set            = 'validation')

#%%
# train for n epochs and save best model
if __name__ == '__main__':
    val_preds = \
        monai_training(train_index      = train_index,
                       val_index        = val_index,
                       train_transforms = train_transforms,
                       val_transforms   = val_transforms,
                       n_epochs         = 10, # must be 10 or more
                       models_dir       = models_dir, 
                       export_val_preds = True)
#%%
val_preds.to_csv(predictions_dir + "val_preds.csv")

"""
Questo salva i pesi dei modelli e le predizioni sul validation nella best epoch.
"""
