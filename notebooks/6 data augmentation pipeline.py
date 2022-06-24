#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:12:59 2022

@author: lab401
"""

"""
Qui scriviamo la pipeline per la data augmentation.
Useremo un compose per:
    Poi vengono applicate casualmente le modifiche di:
        RandAffine per la traslazione
        RandRotate
        RandGaussianNoise
        RandAdjustContrast
    per modificare l'immagine con un set di queste. 
    Viene randomizzata quindi l'applicazione e l'intensità della trasformazione.
"""

from monai.transforms import Compose, LoadImage, SaveImage, RandAffine, RandRotate, RandGaussianNoise, RandAdjustContrast
import pandas as pd
import numpy as np

#%%
"""
Load data:
"""

# Importo l'excel coi metadata delle immagini cropped per prendere la lista completa dei path.
metadata = pd.read_excel("/home/lab401/Analyses/NIFD_exp/data/intermediate/metadata_df.xlsx")

imgs_paths = metadata[(metadata['experimental_group'] == "train")]['path'].tolist()

data_list = [None] * len(imgs_paths)
meta_list = [None] * len(imgs_paths)

for i in range(len(imgs_paths)):

    data_list[i], meta_list[i] = LoadImage()(imgs_paths[i])
    # print(f"image data shape:{data_list[i].shape}")
    # print(f"meta data:{meta_list[i]}")


#%%
"""
Here we perform a data augmentation pipeline:
    Data import
    set transformer for each image
    transform image multiple times by group
"""

# load data has already been performed in a separate chunk.

data_aug_list = []
meta_aug_list = []

for i in range(len(data_list)): # one time per image
    
    # define transformer
    
    aug_transforms = Compose([RandAffine(prob            = 1.0, 
                                         translate_range = (2,2,2),
                                         padding_mode    = "zeros"),
                              RandRotate(prob         = 1.0, 
                                         range_x      = 0.0872665, # è in radianti, equals 5°
                                         range_y      = 0.0,
                                         range_z      = 0.0,
                                         padding_mode = "zeros"),
                              RandGaussianNoise(prob = 0.5,
                                                mean = 0,         # mean activation value in the image
                                                std  = ((np.max(data_list[i]) - np.min(data_list[i]))/100)*2.5)], # 2.5% of range of activation value in the image
                              RandAdjustContrast(prob  = 1.0,
                                                 gamma = (0.0, 3)))  
    
    n_imgs = 5 # così sono 5 immagini aumentate per soggetto
    
    if i % 10 == 0:
        print("Augmenting image " + str(i), " - ", meta_list[i]["filename_or_obj"])
        print("**----**")
        print("Augmenting it ", str(n_imgs), "times --------------------")
    
    # transform image n times
    for ii in range(n_imgs):
        data_aug_list.append(aug_transforms(data_list[i]))
        meta_aug_list.append(meta_list[i])

#%%
"""
Check how many images per group.

    Questa numerosità permette di tenere 450 immagini per il training.
    Per il first training ne useremo 225.
"""

# def condition(x, group):
#     counter = 0
    
#     if x.find(group) != -1:
#         counter += 1
    
#     return counter

# print(sum(condition(x["filename_or_obj"], "g1_ad") for x in meta_aug_list)/6)
# print(sum(condition(x["filename_or_obj"], "g1_ad") for x in meta_aug_list)/6/106)
# print(sum(condition(x["filename_or_obj"], "g2_mci") for x in meta_aug_list)/6)
# print(sum(condition(x["filename_or_obj"], "g2_mci") for x in meta_aug_list)/6/101)
# print(sum(condition(x["filename_or_obj"], "g3_nc") for x in meta_aug_list)/6)
# print(sum(condition(x["filename_or_obj"], "g3_nc") for x in meta_aug_list)/6/62)

#%%
import matplotlib.pyplot as plt


for i in range(data_list[0].shape[2]):

    the_slice = i
    
    plt.figure()
    
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2) 
    
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(data_aug_list[1][:,:,the_slice])
    axarr[1].imshow(data_aug_list[0][:,:,the_slice])
    
    # print("Original shape:", data.shape)
    
    # print("Cropped shape:",data_cr.shape)

#%%
"""
Here we save the augmented images.
"""

# set service variables
num = 1
previous_subj = str()

for i in range(len(data_aug_list)): # one time per augmented image
    
    # print iteration
    if i % 10 == 0:
        print("Saving image " + str(i))
    
    # set current_subj
    current_subj      = meta_aug_list[i]["filename_or_obj"] # è il path
    
    # print("current output:", current_subj)
    # print("previous output:", previous_subj)
    
    # check if the current output_dir is equal to the previous_output_dir
        # update num
    if current_subj == previous_subj:
        num += 1 # if equal, is another aug image of the same subject
    else:
        num = 1  # if not equal, is another subject, so start over by 1
    
    # print("saving with num =", str(num))
    # print("**+++**")
    
    # define saver and use it
    saver = SaveImage(output_dir       = "/".join(meta_aug_list[i]["filename_or_obj"].split("/")[:-1]) + "/",
                      output_postfix   = "aug_" + str(num),
                      output_ext       = "." + ".".join(meta_aug_list[i]["filename_or_obj"].split("/")[-1].split(".")[1:]) ,
                      resample         = False,
                      squeeze_end_dims = True,
                      separate_folder  = False)
    
    # Note: image should be channel-first shape: [C,H,W,[D]].
    
    saver(img       = np.expand_dims(data_aug_list[i], 0), # glielo passo così per aggiungere la singleton dimension 
          meta_data = meta_aug_list[i])
    
    # store previous_output_dir
    previous_subj = current_subj