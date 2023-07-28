import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision import models
from utils import Evaluator
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# Create Custon DataLoader   --------------------------------------------------------------------------------

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
            mask = self.transform(mask)

        return image, mask

# Define data augmentation transformations

data_augmentation = Compose([
    RandomHorizontalFlip(),  # Randomly flip the image horizontally
    RandomVerticalFlip(),    # Randomly flip the image vertically
    RandomRotation(degrees=15),  # Randomly rotate the image up to 15 degrees
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
])

data_transform = Compose([
    #data_augmentation,  # Data augmentation
    ToTensor(),         # Convert images to tensors   
])

# Assuming you have 'train', 'test', and 'validation' folders in your current directory
train_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/train", transform=data_transform)
test_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/test", transform=data_transform)
val_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/validation", transform=data_transform)

# Create DataLoader instances
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create ------------------------------------------------------------------------------------

# Initialize model
model = models.segmentation.deeplabv3_resnet50(pretrained=False)

# Change the last layer
input_of_last_layer = model.classifier[-1].in_channels

# Desired output
num_classes = 1 + 1 # Background class

# Modify last layer
model.classifier[-1] = torch.nn.Conv2d(input_of_last_layer, num_classes, kernel_size=(1, 1))

# Optimizer and Loss -------------------------------------------------------------------------

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Set up metrics
metrics = Evaluator(2)

# Functions for training loop -----------------------------------------------------------------

# Make code agnositc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Defie train step function
def train_step(model,
               data_loader,
               loss_fn,
               optimizer,
               device):

    # Change model to 'train'
    model.train()

    # Move model to device
    model.to(device)

    train_loss  = 0

    # Iterate over all batches of data
    for batch , (X,y) in enumerate(data_loader):

        metrics.reset()
        model.train()
        
        # 0. Send data to GPU
        X, y = X.to(device), y.to(device)        

        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calcualte loss
        loss = loss_fn(y_pred['out'],y.squeeze())
        train_loss += loss.item()
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        
        # Get metrics -------------

        # Get output
        model.eval()
        with torch.inference_mode(): 
            y_pred = model(X)

        # Move prediction to cpu
        y_pred = y_pred['out'].cpu()

        # Make probability prediction
        y_pred = F.softmax(y_pred, dim=1).numpy()
        y_pred = np.argmax(y_pred, axis=1)     

        # Move and squeeze target
        y = y.squeeze().cpu().numpy()        
        metrics.add_batch(y, y_pred)
        
        # Print results
        if batch % 10 == 0:
            # Get metrics on training data        
            Acc_class = metrics.Pixel_Accuracy_Class()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc_class: {:.6f}'.format(
                epoch, batch * len(X), len(data_loader.dataset),
                100. * batch / len(data_loader), loss.item(),Acc_class))
            
         


# Define test step
def test_step(model,
              data_loader,
              loss_fn,
              device):
    
    # Change model to evaluate
    model.to(device)
    model.eval()

    # Reset metrics
    metrics.reset()

    # Iterate over test_data    
    test_loss  = 0
    
    # Run ver data_loader
    for batch, (X, y) in enumerate(data_loader):

        # Move X and y to device
        X, y = X.to(device), y.to(device)

        # Get output
        with torch.inference_mode(): 
            y_pred = model(X)

        # Get test loss
        test_loss += loss_fn(y_pred['out'], y.squeeze()).item()
                
        # Move prediction to cpu
        y_pred = y_pred['out'].cpu()

        # Make probability prediction
        y_pred = F.softmax(y_pred, dim=1).numpy()
        y_pred = np.argmax(y_pred, axis=1)     

        # Move and squeeze target
        y = y.squeeze().cpu().numpy()        

        metrics.add_batch(y, y_pred)
        
    Acc = metrics.Pixel_Accuracy()
    Acc_class = metrics.Pixel_Accuracy_Class()
    mIoU = metrics.Mean_Intersection_over_Union()
    FWIoU = metrics.Frequency_Weighted_Intersection_over_Union()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset)))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)
 


epochs = 3

# Training loop

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")

    train_step(
        model, 
        train_loader, 
        loss_fn,
        optimizer,
        device
    )

    test_step(model,
              test_loader,
              loss_fn,
              device)
    
