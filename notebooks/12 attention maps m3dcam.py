# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:16:45 2022

@author: LAB401

Questo alla fine non ha mai funzionato. Probabilmente inserire m3dcam nel modello cambia qualcosa per cui non basta più la RAM video. Lo abbiamo fatto girare su Colab.
"""

import sys

newpath = "C:/Users/LAB401/Analyses/NIFD_exp/r"
if newpath not in sys.path:
    sys.path.append(newpath)

# Import M3d-CAM
from medcam import medcam
from monai.data import ImageDataset
from torch.utils.data import DataLoader
import torch
import monai
from monai402 import build_data_lists
import pandas as pd

from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, EnsureType
#%%

metadata_df = pd.read_excel("C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/metadata_full.xlsx")

experimental_groups_df = metadata_df[['aug_id', 'experimental_group']]
experimental_groups_df.columns = ['id_aug', 'experimental_group']

resize_ss = 150
# Define transforms for image
test_transforms = Compose([
                           ScaleIntensity(),
                           AddChannel(),
                           Resize(spatial_size = resize_ss,
                                  size_mode = "longest"),
                           EnsureType()])        
#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                 

model = monai.networks.nets.DenseNet121(spatial_dims = 3,
                                        in_channels  = 1,
                                        out_channels = 2)

model.load_state_dict(torch.load("C:/Users/LAB401/Analyses/NIFD_exp/data/models/best_model.pth"
                                  ,
                                  map_location = device
                                 )
                      )

model.to(device = device)
model.eval()

#%%

test_index = build_data_lists(metadata            = metadata_df, 
                              experimental_groups = experimental_groups_df,
                              case_label          = "case",
                              needed_set          = 'test')

# create a validation data loader                                                          ##
test_ds = ImageDataset(image_files = test_index['images'].tolist(), # images[-val_len:]    ##
                       labels      = test_index['labels'].tolist(), # labels[-val_len:]    ##
                       transform   = test_transforms,                                      ##
                       image_only  = False)                                                ##
                                                                                           ##
test_loader = DataLoader(test_ds,                                                          ##
                         batch_size = 1,                                                   ##
                         num_workers = 4,                                                  ##
                         pin_memory = True)                           ##
#%%

# Inject model with M3d-CAM
model = medcam.inject(model,
                      output_dir  = "C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/case_ctrl_ggcam",
                      backend     = 'ggcam',
                      layer       = 'auto',
                      label       = 'best',
                      save_maps   = True,
                      save_scores = True,
                      cudnn       = True)

#%%

subjects = []

if __name__ == '__main__':
    for test_data in test_loader:

        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        test_outputs = model(test_images)
        subjects.append(test_data[2]['filename_or_obj'])
        
"""
SUCA
"""

#%%

# print(subjects)

df = pd.DataFrame(subjects)

# print(df)

df.to_csv("C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/case_ctrl_ggcam/subjects_case_ctrl_ggcam.csv")

#%%
# if __name__ == '__main__':
#       batch = next(iter(test_loader))
#       _ = model(batch[0])

