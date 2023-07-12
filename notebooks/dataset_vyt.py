import shutil
from glob import glob
import os

def move_images(source_dir, dest_dir):
    # Crea la carpeta de destino si no existe
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    ##Images
    images = glob(os.path.join(source_dir, "images_with_gt", "*.tif"))
    
    ##Annotations
    annotations1 = glob(os.path.join(source_dir, "masks", "*.jpeg"))
    for i in range(160):
        shutil.move(images[i], os.path.join(dest_dir, "images_with_gt"))
        shutil.move(annotations1[i], os.path.join(dest_dir, "masks"))


# Directorio fuente que contiene las imágenes
source_dir = "/media/user_home0/juandres/semillero_bcv/hubmap/data/processed/train"

#copy_train(source_dir, dest_copy_train)

# Directorio de destino donde se moverán las imágenes
dest_dir_valid = "/media/user_home0/juandres/semillero_bcv/hubmap/data/processed/validation"
dest_dir_test = "/media/user_home0/juandres/semillero_bcv/hubmap/data/processed/test"


# Llama a la función para mover las imágenes
move_images(source_dir, dest_dir_valid)
move_images(source_dir, dest_dir_test)
