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
print("Training model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ur34.UNetResNet34(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Use with raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# train the model
num_epochs = 10
checkpoint_name = "unet_resnet34"
val_iou, val_dice = ur34t.train_epochs(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, save_checkpoints=True, checkpoint_name=checkpoint_name)

# file_name = f"checkpoints/unet_epoch{num_epochs}_dice{val_dice:.3f}.pth"

# if file_name is not None: torch.save(model.state_dict(), file_name)
