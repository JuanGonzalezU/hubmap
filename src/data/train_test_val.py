# Import librariers
import os
import jsonlines
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd

# Load all the files
all_files_path = os.path.join(os.getcwd().split('src')[0],'data','processed','all_files','images_with_gt')
all_masks_path = os.path.join(os.getcwd().split('src')[0],'data','processed','all_files','masks')

all_images_with_gt = os.listdir(all_files_path)
all_masks = os.listdir(all_masks_path)

# Filter files
filter = True

# Folder to be saved
folder = 'processed_d1'

if filter:

    # Dataset to be used
    dataset = 1

    # Load meta data per file
    tile_meta = pd.read_csv('/home/juandres/semillero_bcv/hubmap/data/raw/tile_meta.csv')

    # Save files
    new_all_images_with_gt = []

    for img in all_images_with_gt:
        
        # Check if file is part of dataset 1
        if tile_meta[tile_meta['dataset']==dataset]['id'].str.contains(img.split('.tif')[0]).sum() == 1:
            new_all_images_with_gt.append(img)            

        
# Replace images to be used
all_images_with_gt = new_all_images_with_gt


# Create a trian test split on the files

def train_test_val_split(data, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):

    # Split the data into train and remaining data
    train_data, remaining_data = train_test_split(data, test_size=(1 - train_ratio), random_state=57)
    
    # Split the remaining data into test and validation sets
    test_data, val_data = train_test_split(remaining_data, test_size=(val_ratio / (test_ratio + val_ratio)), random_state=57)
    
    return train_data, test_data, val_data

train_data, test_data, val_data = train_test_val_split(all_images_with_gt)

# Create train test and val folders if they don't exists
data_path = os.path.join(os.getcwd().split('src')[0],'data',folder)
source_path = os.path.join(os.getcwd().split('src')[0],'data','processed')

# Directories to be created
all_dirs = ['train','test','validation']

for dir in all_dirs:
    dir_path = os.path.join(data_path,dir)
    if os.path.exists(dir_path):
        pass        
    else:
        os.makedirs(dir_path)


# Iterarte over all directories and create subfolders

# Directories to be created
all_dirs = ['train','test','validation']
sub_dirs = ['images_with_gt','masks']

for dir in all_dirs:    
    for sub_dir in sub_dirs:
        dir_path = os.path.join(data_path,dir,sub_dir)
        if os.path.exists(dir_path):
            pass        
        else:
            os.makedirs(dir_path)

# Copy files
groups_data = [train_data,test_data,val_data]

i = 0
for group in groups_data:

    # Iterate over all files in the group
    for file in group:

        # Find equivalent file in masks
        name = file.split('.')[0]

        # Find mask element
        for mask_element in all_masks:       

            # Check if file and masks are the same     
            if mask_element.split('.')[0] == name:

                # Copy both images
                source_path_image = os.path.join(source_path,'all_files','images_with_gt',file)
                source_path_mask = os.path.join(source_path,'all_files','masks',mask_element)
                
                dest_path_image = os.path.join(data_path,all_dirs[i],'images_with_gt',file)
                dest_path_mask = os.path.join(data_path,all_dirs[i],'masks',mask_element)

                
                shutil.copy(source_path_image,dest_path_image)
                shutil.copy(source_path_mask,dest_path_mask)
                
        
    i+=1           
