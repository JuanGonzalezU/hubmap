# Import librariers
import os
import pandas as pd
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shutil
from PIL import Image
from tqdm import tqdm

# Functions for flood fill algorithm
from queue import Queue

raw_data_path = os.path.join(os.getcwd().split('src')[0],'data','raw')
content = os.listdir(raw_data_path)

# Function for file with data path
join_path = lambda path1 : os.path.join(raw_data_path,path1)


def get_annotations(id):
    # Open the JSONL file

    with jsonlines.open(join_path('polygons.jsonl')) as reader:
        # Iterate over each line in the file    
        ids = []    
        for line in reader:
            # Process each JSON object
            if id == 'get_values':     
                ids.append(line['id'])                   

            if line['id'] == id:
                return pd.DataFrame(line)
            
        return ids

def swap_lower_upper_triangle(matrix):
    rows = len(matrix)

    for i in range(rows):
        for j in range(i+1, rows):
            # Swap elements in lower and upper triangle
            matrix[i, j], matrix[j, i] = matrix[j, i], matrix[i, j]

    return matrix

def flood_fill2(mask, x, y, new_color):
    w = mask.shape[0]
    h = mask.shape[1]
    old_color = mask[x, y]
    
    if old_color == new_color:
        # No need to fill if the new color is the same as the old color
        return mask
    
    queue = Queue()
    queue.put((x, y))
    
    while not queue.empty():
        x, y = queue.get()
        
        if x < 0 or x >= w or y < 0 or y >= h or mask[x, y] != old_color:
            continue
        
        mask[x, y] = new_color
        
        queue.put((x + 1, y))
        queue.put((x - 1, y))
        queue.put((x, y + 1))
        queue.put((x, y - 1))
    
    return mask

# Convert coordinates to an image

# Mask will be a 3d matrix. Shape will be the same as input image, and the 3d dimension will contain the masks of each class.

# Get names of files with annotations
names = get_annotations('get_values')

# Path for the masks
mask_path = os.path.join(os.getcwd().split('src')[0],'data/processed/all_files/masks1')

#mask_path = os.path.join(os.getcwd().split('hubmap')[0],'test/masks/')

for count in tqdm(range(len(names))):

    # Convert coordinates to an image

    # Get name of file (same as ID)
    id = names[count]

    # Get annotations
    annotations = get_annotations(id)

    # Read on Show image
    img = plt.imread(join_path('train/'+id+'.tif'))

    # Create mask (m,n)
    all_mask = np.zeros(img.shape[:2])

    # Iterate over all elements in annotation data frame
    for index, row in annotations.iterrows():
        
        # Mask per element
        mask = np.zeros(img.shape[:2])

        # Get row anonotations
        tmp_row_ann = row['annotations']

        # Get type of the element
        tmp_type = tmp_row_ann['type']

        # Get color depending on the type
        if tmp_type != 'blood_vessel':
            continue    
   
        # Get coordinates
        tmp_coords = np.array(tmp_row_ann['coordinates']).squeeze()

        # Add to the mask the polygon depending on the class
        for coord in tmp_coords: 
            mask[coord[0],coord[1]] = 1         

        # Select seed for flood fill
        tmp_x_max = max(tmp_coords[:,0])
        tmp_x_min = min(tmp_coords[:,0])
        x_range = np.arange(tmp_x_min,tmp_x_max)

        tmp_y_max = max(tmp_coords[:,1])
        tmp_y_min = min(tmp_coords[:,1])
        y_range = np.arange(tmp_y_min,tmp_y_max)

        # Get random point around borders
        random_initial_seed = [np.random.choice(x_range),np.random.choice(y_range)]
        random_initial_seed_p = Point(random_initial_seed)

        # Create polygon
        polygon = Polygon(tmp_coords)
        n=0
        while polygon.contains(random_initial_seed_p) == False:    
            #print(polygon.contains(random_initial_seed_p))    
            random_initial_seed = [np.random.choice(x_range),np.random.choice(y_range)]
            random_initial_seed_p = Point(random_initial_seed)    
            n+=1
            # If there's nopoint inside the polygon, break the loop
            if n>50:
                break    
        if n>50:
            continue

        # Make flood fill    
        flood_fill2(mask,random_initial_seed[0],random_initial_seed[1],1)

        # Add region to mask
        all_mask += mask
            
    # Correct mask 
    all_mask = swap_lower_upper_triangle(all_mask)      
    all_mask[all_mask >= 1] = 255 

    # Create an RGB mask
    RGB_mask = np.zeros([512, 512, 3])

    # Binary mask will be saved on R channel (dim = 0, pos = 0)
    RGB_mask[:,:,0] = all_mask
    RGB_mask = RGB_mask.astype(np.uint8)

    # Convert matrix to image and save it
    im = Image.fromarray(RGB_mask)
    im.save(mask_path + "/{}.jpeg".format(id))

