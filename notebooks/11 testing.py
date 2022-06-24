#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:10:59 2022

@author: lab401
"""

import sys
sys.path.append("C:/Users/LAB401/Analyses/NIFD_exp/r/")

from monai.transforms import AddChannel, Compose, ScaleIntensity, EnsureType, Resize#, SqueezeDim, 

import pandas as pd
# from pandas import ExcelWriter

from monai402 import monai_test, build_data_lists
#%%

resize_ss = 150

metadata_df = pd.read_excel('C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/metadata_full.xlsx')

experimental_groups_df = metadata_df[['aug_id', 'experimental_group']]
experimental_groups_df.columns = ['id_aug', 'experimental_group']

models_dir = 'C:/Users/LAB401/Analyses/NIFD_exp/data/models/'
predictions_dir = 'C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/MONAI predictions/'

# Define transforms for image
test_transforms = Compose([
                           ScaleIntensity(),
                           AddChannel(),
                           Resize(spatial_size = resize_ss,
                                  size_mode = "longest"),
                           EnsureType()])                        
        
test_index = build_data_lists(metadata            = metadata_df, 
                              experimental_groups = experimental_groups_df,
                              case_label          = "case",
                              needed_set          = 'test')
#%%
if __name__ == '__main__':
    preds = \
        monai_test(test_index = test_index,
                   test_transforms = test_transforms,
                   model_weights = models_dir + \
                                   "best_model.pth")

#%%

# preds.to_csv(predictions_dir + "test_preds.csv")

        
        
        