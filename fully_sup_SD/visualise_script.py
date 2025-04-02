import sys
import os

# Add the folder containing unet_resnet34_class.py to sys.path
module_path = os.path.abspath(os.path.join(os.getcwd(), "fully_sup_SD"))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

import unet_resnet34_class as ur34
import petseg_dataset as psd
import unet_resnet34_train as ur34t

import visualise_predictions as vp

random_seed = 1
torch.manual_seed(random_seed)



# For input images (3-channel RGB)
image_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# For masks (1-channel, grayscale)
# mask_transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),  # Converts to [1, H, W] and scales to [0, 1]
# ])

print("Loading dataset...")

# Create dataset instance
dataset = psd.PetSegmentationDataset(
    image_dir="../oxford-iiit-pet/images",
    mask_dir="../oxford-iiit-pet/annotations/trimaps",
    image_transform=image_transform,
    # mask_transform=mask_transform
    binarize=True
)
print(f"Loaded {len(dataset)} images and masks.")

# Split into train and val (e.g. 80/20)
print("Splitting dataset into train and validation sets...")
train_split = 0.8
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create dataloaders
print("Creating dataloaders...")
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# create the model and training objects
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_names = []
file_names.append("checkpoints/unet_epoch0_dice0.876.pth")
# file_names.append("checkpoints/unet_epoch1_dice0.891.pth")
# file_names.append("checkpoints/unet_epoch2_dice0.905.pth")
# file_names.append("checkpoints/unet_epoch3_dice0.905.pth")
# file_names.append("checkpoints/unet_epoch4_dice0.910.pth")
# file_names.append("checkpoints/unet_epoch5_dice0.915.pth")
# file_names.append("checkpoints/unet_epoch6_dice0.913.pth")
# file_names.append("checkpoints/unet_epoch7_dice0.919.pth")
# file_names.append("checkpoints/unet_epoch8_dice0.915.pth")
file_names.append("checkpoints/unet_epoch9_dice0.917.pth")

model = ur34.UNetResNet34(num_classes=1)  # or your model class

for file_name in file_names:
    
    model.load_state_dict(torch.load(file_name))
    model.to(device)
    model.eval()  # if you're using it for inference

    print(f"Loaded model from {file_name}")
    vp.visualise_predictions(model, val_loader, device, num_images=1, threshold=0.5)

