import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor,Compose


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
    


data_transform = Compose([
    ToTensor(),         # Convert images to tensors   
])


# Assuming you have 'train', 'test', and 'validation' folders in your current directory
train_dataset = CustomDataset("/home/juandres/semillero_bcv/hubmap/data/processed/train",transform=data_transform)

# Create DataLoader instances
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6)

# Get mean and std
mean = 0.
std = 0.
for images, _ in train_loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(train_loader.dataset)
std /= len(train_loader.dataset)
torch.set_printoptions(precision=10)
print(mean,std)