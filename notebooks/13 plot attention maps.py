# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:05:19 2022

@author: LAB401
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, EnsureType
from torch.utils.data import DataLoader

import nibabel as nib
from scipy.stats import iqr

#%%
def get_stds(img_arr):
    """
    Here we calculate the standard deviation for each layer of the att map.
    Then we extract the layers with the highest std for each perspective.
    """
    
    uno = []
    for i in range(len(img_arr)):
        uno.append(np.std(img_arr[i,:,:]))

    due = []
    for i in range(len(img_arr[0])):
        due.append(np.std(img_arr[:,i,:]))
        
    tre = []
    for i in range(len(img_arr[0][0])):
        tre.append(np.std(img_arr[:,:,i]))

    stds = [pd.to_numeric(uno).argmax(),
            pd.to_numeric(due).argmax(),
            pd.to_numeric(tre).argmax()]
    
    return uno, due, tre, stds
#%%

def get_iqrs(img_arr):
    """
    Here we calculate the standard deviation for each layer of the att map.
    Then we extract the layers with the highest std for each perspective.
    """
    
    uno = []
    for i in range(len(img_arr)):
        uno.append(iqr(img_arr[i,:,:]))

    due = []
    for i in range(len(img_arr[0])):
        due.append(iqr(img_arr[:,i,:]))
        
    tre = []
    for i in range(len(img_arr[0][0])):
        tre.append(iqr(img_arr[:,:,i]))

    iqrs = [pd.to_numeric(uno).argmax(),
            pd.to_numeric(due).argmax(),
            pd.to_numeric(tre).argmax()]
    
    return uno, due, tre, iqrs

#%%
df = pd.read_csv("C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/subjects_case_ctrl_ggcam.csv").rename(columns={"Unnamed: 0" : "subject",
                                                                                                                                "0" : "path"})
df['path'][0] = "C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/collection1_first_t1mprage_t1_linear/subjects/sub-NIFD1S0040/ses-M00/t1_linear/sub-NIFD1S0040_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
df['class'] = "case"

resize_ss = 150
# Define transforms for image
test_transforms = Compose([
                           ScaleIntensity(),
                           AddChannel(),
                           Resize(spatial_size = resize_ss,
                                  size_mode = "longest"),
                           EnsureType()])      
#%%
test_ds = ImageDataset(image_files = df['path'], # images[-val_len:]                       
                       labels      = df['class'], # labels[-val_len:]                      
                       transform   = test_transforms,                                      
                       image_only  = False)                                                


test_loader = DataLoader(test_ds,                                                          
                         batch_size = 1,                                                   
                         num_workers = 4,                                                  
                         pin_memory = True)                                                

case_index = 0

i = 0

if __name__ == '__main__':
    for test_images in test_loader:
    
        print("sei nel giro" + str(i))
        
        case_mri = test_images
        
        if i == case_index:
            print("sei nell'if! " + str(i) + " " + str(case_index))
            break
        
        i += 1
        
#%%

df['path'][0] = "C:/Users/LAB401/Analyses/NIFD_exp/data/intermediate/collection1_first_t1mprage_t1_linear/subjects/sub-NIFD1S0182/ses-M00/t1_linear/sub-NIFD1S0182_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
df['class'] = "ctrl"

test_ds = ImageDataset(image_files = df['path'], # images[-val_len:]                       
                       labels      = df['class'], # labels[-val_len:]                      
                       transform   = test_transforms,                                      
                       image_only  = False)                                                


test_loader = DataLoader(test_ds,                                                          
                         batch_size = 1,                                                   
                         num_workers = 4,                                                  
                         pin_memory = True)                                                

ctrl_index = 0

i = 0

if __name__ == '__main__':
    for test_images in test_loader:
    
        print("sei nel giro" + str(i))
        
        ctrl_mri = test_images
        
        if i == ctrl_index:
            print("sei nell'if! " + str(i) + " " + str(case_index))
            break
        
        i += 1

#%%

case_img = nib.load("C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/attention_map_0_0_0_case.nii.gz")
case_arr = np.array(case_img.dataobj)


ctrl_img = nib.load("C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/attention_map_0_0_0_ctrl.nii.gz")
ctrl_arr = np.array(ctrl_img.dataobj)
#%%
"""
Get standard deviations:
"""

case_stds_uno, case_stds_due, case_stds_tre, case_stds = get_stds(case_arr)
case_iqrs_uno, case_iqrs_due, case_iqrs_tre, case_iqrs = get_iqrs(case_arr)

ctrl_stds_uno, ctrl_stds_due, ctrl_stds_tre, ctrl_stds = get_stds(ctrl_arr)
ctrl_iqrs_uno, ctrl_iqrs_due, ctrl_iqrs_tre, ctrl_iqrs = get_iqrs(ctrl_arr)

#%%
"""
Versione statica
"""
plt.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,
                        3
                        ) 
plt.subplots_adjust(wspace= 0.25, hspace= 0)

#-------------------------------------
# CASE
#-------------------------------------

axarr[0,0].imshow(np.flip(case_mri[0][0][0][
                                        :,
                                        case_iqrs[0], 
                                        :
                                            ].detach().numpy(),1).transpose(),
                  cmap = 'Greys_r',
                  alpha = 1,
                  norm = matplotlib.colors.Normalize())
axarr[1,0].imshow(np.flip(case_arr[
                            case_iqrs[0],
                            :,
                            :
                            ],0),
                  cmap = 'Greys',
                  alpha = 1)

axarr[0,0].set_xticks([])
axarr[0,0].set_yticks([])
axarr[0,0].axis('off')
axarr[1,0].set_xticks([])
axarr[1,0].set_yticks([])
axarr[1,0].axis('off')

axarr[0,1].imshow(np.flip(case_mri[0][0][0][
                                          :,
                                          :,
                                          case_iqrs[1]
                                          ].detach().numpy().transpose(),0),
                  cmap = 'Greys_r',
                  alpha = 1,
                  norm = matplotlib.colors.Normalize())
axarr[1,1].imshow(np.flip(case_arr[
                                    :,
                                    case_iqrs[1],
                                    : 
                                    ],
                          0),
                  cmap = "Greys",
                  alpha = 1)

axarr[0,1].set_xticks([])
axarr[0,1].set_yticks([])
axarr[0,1].axis('off')
axarr[1,1].set_xticks([])
axarr[1,1].set_yticks([])
axarr[1,1].axis('off')


axarr[0,2].imshow(np.flip(case_mri[0][0][0][
                                                  case_iqrs[2],
                                                  :,
                                                  :
                                                  ].detach().numpy().transpose(),0),
                  cmap = 'Greys_r',
                  alpha = 1,
                  norm = matplotlib.colors.Normalize())
axarr[1,2].imshow(np.flip(case_arr[
                                    :,
                                    :,
                                    case_iqrs[2]
                                    ].transpose(),0),
                  cmap = "Greys",
                  alpha = 1)

axarr[0,2].set_xticks([])
axarr[0,2].set_yticks([])
axarr[0,2].axis('off')
axarr[1,2].set_xticks([])
axarr[1,2].set_yticks([])
axarr[1,2].axis('off')

#%%
"""
salva tutte le immagini per costruire una gif (in r)
"""


for i in range(150):
    plt.figure()
    
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2,
                            2
                            ) 
    
    plt.subplots_adjust(wspace = 0.25,
                        hspace= 0)
    
    actual_slice = 149-i
    
    #-------------------------------------
    # CASE
    #-------------------------------------
    
    axarr[0,0].imshow(np.flip(case_mri[0][0][0][
                                            :,
                                            actual_slice, 
                                            :
                                                ].detach().numpy(),1).transpose(),
                      cmap = 'Greys_r',
                      alpha = 1,
                      norm = matplotlib.colors.Normalize())
    axarr[1,0].imshow(np.flip(case_arr[
                                actual_slice,
                                :,
                                :
                                ],0),
                      cmap = 'Greys',
                      alpha = 1)
    
    
    #-------------------------------------
    # CTRL
    #-------------------------------------
    
    axarr[0,1].imshow(np.flip(ctrl_mri[0][0][0][
                                            :,
                                            actual_slice, 
                                            :
                                                ].detach().numpy(),1).transpose(),
                      cmap = 'Greys_r',
                      alpha = 1,
                      norm = matplotlib.colors.Normalize())
    axarr[1,1].imshow(np.flip(ctrl_arr[
                                actual_slice,
                                :,
                                :
                                ],0),
                      cmap = 'Greys',
                      alpha = 1)
    
        
    axarr[0,0].set_title("FTD")
    axarr[0,1].set_title("NC")

    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])
    axarr[0,0].axis('off')
    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])
    axarr[1,0].axis('off')
    axarr[0,1].set_xticks([])
    axarr[0,1].set_yticks([])
    axarr[0,1].axis('off')
    axarr[1,1].set_xticks([])
    axarr[1,1].set_yticks([])
    axarr[1,1].axis('off')
    
    plt.savefig('C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/gif_files/img' + str(i) + '.png',
                bbox_inches='tight', 
                dpi = 600)
    
#%%
"""
salva i singoli plot
"""

slice_to_plot = 59

plt.imshow(np.flip(case_mri[0][0][0][
                                        :,
                                        slice_to_plot, 
                                        :
                                            ].detach().numpy(),1).transpose(),
                  cmap = 'Greys_r',
                  alpha = 1,
                  norm = matplotlib.colors.Normalize())
    
plt.axis('off')

plt.savefig('C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/case_mri.png',
            bbox_inches='tight',
            dpi = 600)

plt.imshow(np.flip(case_arr[
                            slice_to_plot,
                            :,
                            :
                            ],0),
                  cmap = 'Greys',
                  alpha = 1)

plt.axis('off')
    
plt.savefig('C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/case_am.png',
            bbox_inches='tight',
            dpi = 600)

plt.imshow(np.flip(ctrl_mri[0][0][0][
                                        :,
                                        slice_to_plot, 
                                        :
                                            ].detach().numpy(),1).transpose(),
                  cmap = 'Greys_r',
                  alpha = 1,
                  norm = matplotlib.colors.Normalize())
    
plt.axis('off')

plt.savefig('C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/ctrl_mri.png',
            bbox_inches='tight',
            dpi = 600)

plt.imshow(np.flip(ctrl_arr[
                            slice_to_plot,
                            :,
                            :
                            ],0),
                  cmap = 'Greys',
                  alpha = 1)

plt.axis('off')
    
plt.savefig('C:/Users/LAB401/Analyses/NIFD_exp/data/attention_maps/ctrl_am.png',
            bbox_inches='tight',
            dpi = 600)
