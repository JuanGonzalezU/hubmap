# This scipt creates one folde per model under data/model_outputs with the predicted masks of validation
# folder. If the folders doesn't exists, it creates a new one. If not, it goes to the next model. 


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np  
from torchvision import models
from PIL import Image

# Class for a custom dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = "images_with_gt"
        self.mask_folder = "masks"
        self.image_filenames = os.listdir(os.path.join(self.root_dir, self.image_folder))
        self.mask_filenames = os.listdir(os.path.join(self.root_dir, self.mask_folder))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_base, _ = os.path.splitext(image_name)  # Extract base name without extension

        mask_name = [mask for mask in self.mask_filenames if mask.startswith(image_base)][0]
        mask_base, _ = os.path.splitext(mask_name)  # Extract base name without extension

        image_path = os.path.join(self.root_dir, self.image_folder, image_name)
        mask_path = os.path.join(self.root_dir, self.mask_folder, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Get only the RED channel of the mask
        mask = np.array(mask)
        mask = (mask[:,:,0]/255).astype(int)     

        if self.transform is not None:
            image = self.transform(image)
            #mask = self.transform(mask)

        return image, mask, mask_name
    
# Class for printing in colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Transfomrations

# Set device for predictions
device = 'cuda'

# All data transforms

data_transform0 = Compose([
    ToTensor()         # Convert images to tensors   
])

mean_v1 = [0.485, 0.456, 0.406]
std_v1 = [0.229, 0.224, 0.225]

data_transform1 = Compose([
    ToTensor(),         # Convert images to tensors   
    Normalize(mean_v1, std_v1)
])

# Values of current dataset
mean_v2 = [0.6300351620, 0.4150927663, 0.6850880980]
std_v2 = [0.1454657167, 0.1958113164, 0.1234567240]

data_transform2 = Compose([
    ToTensor(),         # Convert images to tensors   
    Normalize(mean_v2, std_v2)
])

all_data_transfroms  = [data_transform0,data_transform1,data_transform2]

# Define Batch Size
batch_size = 8
num_workers = 5

# Iterate over all the models contained models directry

# Get files
all_models = []

# Set the parameters for each one of the models. This should be updated when a new model 
# is trained

# Model_name, transformation, pre_tained (true or false),'Model used'

models_parameters = [['model_Res50_norm_own',all_data_transfroms[2],False,'RESNET50'],
                     ['model_Res50_w_norm',all_data_transfroms[1],True,'RESNET50'],
                     ['model_Res50_wt_transform',all_data_transfroms[0],True,'RESNET50'],
                     ['model_Res101_wt_transform',all_data_transfroms[0],True,'RESNET101'],
                     ['model_RES50_norm_own_30',all_data_transfroms[2],True,'RESNET50'],
                     ['model_Res50_FCN',all_data_transfroms[2],True,'RESNETFCN']]

# Check all files are .pt

for num_model, model_name in enumerate(os.listdir('/home/juandres/semillero_bcv/hubmap/models/')):    

    # Look for .pt files
    if model_name.endswith('.pt'):
        print('------------------------------------')
        print(num_model,'- Model:', model_name)

        # Get model name
        name = model_name.split('.pt')[0]

        # Create directory
        folder_path = os.path.join('/home/juandres/semillero_bcv/hubmap/data/models_output',name)

        # Check if it already exists
        if not os.path.exists(folder_path):

            # Create folder
            os.mkdir(folder_path)
            print(bcolors.OKGREEN+'Folder Created!'+bcolors.ENDC)

            # Get model details

        else:
            print(bcolors.WARNING+'Folder Already Created'+bcolors.ENDC)
            
        # Check if the directory is empty
        files_folder = os.listdir(folder_path)
        
        if len(files_folder)>0:
            print(bcolors.WARNING+'Predictions Already Created'+bcolors.ENDC)
            continue

        # Get model details      
        flag = False  
        for i,parameter in enumerate(models_parameters):   
            if parameter[0] == name:
                target_parameter = parameter
                print(bcolors.OKGREEN+'Model Training Parameters found!'+bcolors.ENDC)
                break
            elif (parameter[0] != name) and (i == len(models_parameters)-1):
                print(bcolors.FAIL +'No parameters found, add them! ... moving to next model'+bcolors.ENDC)
                flag = True
        if flag:
            continue

        # Create custom dataset
        val_dataset_transform = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/validation",transform=parameter[1])
        #
        # Create dataloader
        val_loader_tansform = DataLoader(val_dataset_transform, batch_size=batch_size, shuffle=False,num_workers=num_workers)
        print('── Validation Data Loaded')
         
        # Create model
        if parameter[3] == 'RESNET50':
            model = models.segmentation.deeplabv3_resnet50(pretrained=parameter[2])
        elif parameter[3] == 'RESNET101':
            model = models.segmentation.deeplabv3_resnet101(pretrained=parameter[2])
        elif parameter[3] == 'RESNETFCN':
            model = models.segmentation.fcn_resnet50(pretrained=parameter[2])
        else:
            print('No model created, check the models_parameters ... moving to next model') 
            flag = True

        if flag:
            continue        
                                
        # Change the last layer
        input_of_last_layer = model.classifier[-1].in_channels    
        
        # Modify last layer
        model.classifier[-1] = torch.nn.Conv2d(input_of_last_layer, 2, kernel_size=(1, 1))
        
        print('── Base Model created')    

        # Import weights
        trained_model = torch.load(os.path.join('/home/juandres/semillero_bcv/hubmap/models/',model_name))
        
        # Move weights into crated model
        model.load_state_dict(trained_model)   
        
        print('── Weights Correctly Imported')      

        # Make predictions

        # Turn model into eval method
        model.eval()

        # Iterate over all batches in validation
        for batch , (X,y,file_name) in enumerate(val_loader_tansform):
            
            # Move model to device
            X,y,model = X.to(device), y.to(device), model.to(device)

            # Make predictions
            with torch.inference_mode(): 
                y_pred = model(X)
            
            # Convert to prediction 
            y_pred = torch.argmax(y_pred['out'],dim = 1)

            # Iterate over all images and save it to path
            for img_count in range(0,y_pred.shape[0]):

                # Convert to numpy array 
                img = (255*y_pred[img_count,:,:].to('cpu').numpy()).astype(np.uint8)

                # Create empty RGB image
                #RGB_img = np.zeros([img.shape[0],img.shape[1],3])
                RGB_img = np.zeros([512, 512, 3]).astype(np.uint8)

                # Replace red channel with predicted mask
                RGB_img[:,:,0] = img

                # Create path for saving image
                path_img = '/home/juandres/semillero_bcv/hubmap/data/models_output'
                
                path_img = os.path.join(path_img,folder_path,file_name[img_count])

                # Save image
                Image.fromarray(RGB_img).save(path_img)

            print('  |── Batch : ',batch ,' - Images saved!')

