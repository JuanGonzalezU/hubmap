# Import librariers
import os
import pandas as pd
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import seaborn as sns
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


def dfs(grid,i,j,old_color, new_color):
    n = grid.shape[0]
    m = grid.shape[1]

    if i < 0 or i >= n or j < 0 or j >= m or grid[i,j] != old_color:
        return
    else:
        grid[i,j] = new_color
        dfs(grid, i+1, j, old_color, new_color)
        dfs(grid, i-1, j, old_color, new_color)
        dfs(grid, i, j+1, old_color, new_color)
        dfs(grid, i, j-1, old_color, new_color)

def flood_fill(grid,i,j,new_color):
    old_color = grid[i,j]
    if old_color == new_color:
        return
    dfs(grid,i,j,old_color,new_color)

def flood_fill1(mask,x,y,new_color):
    w = mask.shape[0]
    h = mask.shape[1]
    old_color = mask[x,y]
    queue = Queue()
    queue.put((x,y))
    while not queue.empty():
        x,y = queue.get()
        if x < 0 or x >= w or y < 0 or y >= h or mask[x,y] != old_color:
            continue
        else:
            mask[x,y] = new_color
            queue.put((x+1,y))
            queue.put((x-1,y))
            queue.put((x,y+1))
            queue.put((x,y-1))

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
mask_path = os.path.join(os.getcwd().split('src')[0],'data/processed/train/masks/')
#mask_path = os.path.join(os.getcwd().split('hubmap')[0],'test/masks/')

for count in tqdm(range(len(names))):

    # Convert coordinates to an image

    # Mask will be a 3d matrix. Shape will be the same as input image, and the 3d dimension will contain the masks of each class.

    # Get name of file (same as ID)
    id = names[count]

    # Get annotations
    annotations = get_annotations(id)

    # Read on Show image
    img = plt.imread(join_path('train/'+id+'.tif'))

    # Create mask (m,n,classes) - they are 3, blood vessel, glomerulus and undetermined 
    mask = np.zeros(img.shape)

    # Iterate over all elements in annotation data frame
    for index, row in annotations.iterrows():
        # Get row anonotations
        tmp_row_ann = row['annotations']

        # Get type of the element
        tmp_type = tmp_row_ann['type']

        # Get color depending on the type
        if tmp_type == 'blood_vessel':
            color = 'red'
            chann_mask = 0
        elif tmp_type == 'glomerulus':
            color = 'green'
            chann_mask = 1
        else:
            color = 'blue'
            chann_mask = 2

        # Get coordinates
        tmp_coords = np.array(tmp_row_ann['coordinates']).squeeze()

        # Add to the mask the polygon depending on the class
        for coord in tmp_coords: 
            mask[coord[0],coord[1],chann_mask] = 1        

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
    
        if n<50:
            flood_fill2(mask[:,:,chann_mask],random_initial_seed[0],random_initial_seed[1],1)
            

        # Correct mask 
    mask[:,:,0] = swap_lower_upper_triangle(mask[:,:,0])  
    mask[:,:,1] = swap_lower_upper_triangle(mask[:,:,1])  
    mask[:,:,2] = swap_lower_upper_triangle(mask[:,:,2])  

        #breakpoint()

    plt.imsave(mask_path + "{}.jpeg".format(id), mask) 

    """
    mask = mask*255
    mask = mask.astype(np.uint8)

    im = Image.fromarray(mask)
    im.save(mask_path + "{}.tif".format(id))
    """
                
