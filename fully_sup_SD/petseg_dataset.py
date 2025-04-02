import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class PetSegmentationDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, image_transform=None, resize_size=(256, 256), binarize=False):
        self.image_paths = sorted([p for p in Path(image_dir).glob("*.jpg")])
        self.mask_paths = sorted([p for p in Path(mask_dir).glob("*.png")])
        self.image_transform = image_transform
        self.resize_size = resize_size
        self.binarize = binarize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # Load mask as raw label array
        mask = Image.open(self.mask_paths[idx])
        mask = mask.resize(self.resize_size, Image.NEAREST)
        mask = np.array(mask)

        if self.binarize:
            # 1 = pet, 0 = everything else
            mask = (mask == 1).astype(np.float32)
        else:
            # 1 = pet, 2 = background, 3 = border — preserve as is
            mask = mask.astype(np.int64)

        mask = torch.from_numpy(mask).unsqueeze(0)  # shape: [1, H, W]

        return image, mask