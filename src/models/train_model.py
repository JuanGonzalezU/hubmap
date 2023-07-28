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
            #mask = self.transform(mask)

        return image, mask

# Define data augmentation transformations

data_augmentation = Compose([
    RandomHorizontalFlip(),  # Randomly flip the image horizontally
    RandomVerticalFlip(),    # Randomly flip the image vertically
    RandomRotation(degrees=15),  # Randomly rotate the image up to 15 degrees
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = Compose([
    data_augmentation,  # Data augmentation
    ToTensor(),         # Convert images to tensors   
    Normalize(mean, std)
])

# Assuming you have 'train', 'test', and 'validation' folders in your current directory
train_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/train", transform=data_transform)
test_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/test", transform=data_transform)
val_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/validation", transform=data_transform)

# Create DataLoader instances
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=6)

# Create ------------------------------------------------------------------------------------

# Initialize model
model = models.segmentation.deeplabv3_resnet50(pretrained=False)

# Change the last layer
input_of_last_layer = model.classifier[-1].in_channels

# Desired output
num_classes = 1 

# Modify last layer
model.classifier[-1] = torch.nn.Conv2d(input_of_last_layer, num_classes, kernel_size=(1, 1))

# Optimizer and Loss -------------------------------------------------------------------------

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# Set up metrics
metrics = Evaluator(2)

# Metrics functions ---------------------------------------------------------------------------

def pixel_accuracy_single(y_pred, y_true):
    y_pred = y_pred.float()  # Convert to float to avoid integer division
    correct_pixels = torch.sum(y_pred == y_true)
    total_pixels = y_true.numel()
    accuracy = correct_pixels.float() / total_pixels
    return accuracy


def intersection_over_union_single(y_pred, y_true):
    y_pred = y_pred.float()  # Convert to float to avoid integer division
    intersection = torch.logical_and(y_pred, y_true)
    union = torch.logical_or(y_pred, y_true)
    
    iou = torch.sum(intersection).float() / (torch.sum(union).float() + 1e-8)  # Add epsilon to avoid divide by zero
    return iou


def class_accuracy_single(y_pred, y_true, num_classes):
    y_pred = y_pred.float()  # Convert to float to avoid integer comparison
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for class_idx in range(num_classes):
        class_mask = (y_true == class_idx)
        correct_predictions = torch.sum(y_pred[class_mask] == y_true[class_mask])
        total_pixels = torch.sum(class_mask)

        class_correct[class_idx] = correct_predictions
        class_total[class_idx] = total_pixels

    class_accuracy = class_correct.float() / (class_total.float() + 1e-8)  # Add epsilon to avoid divide by zero
    return class_accuracy


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
    model.train()

    # Iterate over all batches of data
    for batch , (X,y) in enumerate(data_loader):

        # 0. Send data to GPU
        X, y = X.to(device), y.to(device).long().squeeze_(1)        

        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calcualte loss      
        loss = loss_fn(y_pred['out'].squeeze(), y.float())

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        
        # Get metrics -------------
        pa = pixel_accuracy_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:]).item()
        iou = intersection_over_union_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:]).item()
        ca = class_accuracy_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:],2)

        # Print results
        if batch % 1 == 0:
            # Get metrics on training data        
            Acc_class = metrics.Pixel_Accuracy_Class()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc1 : {:.6f} Acc2 : {:.6f} IoU : {:.6f} pAcc : {:.6f}'.format(
                epoch, batch * len(X), len(data_loader.dataset),
                100. * batch / len(data_loader), loss.item(),ca[0].item(),ca[1].item(),iou,pa))
            
         


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
    pa,iou,ca = 0,0,0
    for batch, (X, y) in enumerate(data_loader):

        # Move X and y to device
        X, y = X.to(device), y.to(device)

        # Get output
        with torch.inference_mode(): 
            y_pred = model(X)

         # Get metrics -------------
        pa += pixel_accuracy_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:]).item()
        iou += intersection_over_union_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:]).item()
        ca += class_accuracy_single(torch.argmax(y_pred['out'],dim = 1)[1,:,:],y.squeeze()[1,:,:],2)

    pa = pa/batch
    iou = iou/batch
    ca = ca/batch
    
    print('Acc1 : {:.6f} Acc2 : {:.6f} IoU : {:.6f} pAcc : {:.6f}'.format(
        ca[0].item(),ca[1].item(),iou,pa))

epochs = 15

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
    
