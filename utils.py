#Load and examine the masks with the origional image
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#customized dataset class for imagenet and mask
class ImageNetandMask(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.samples = []  # A list of tuples (image_path, mask_path, class_index)
        self.class_names = []

        # Assuming same structure for both image_dir and mask_dir
        for class_name in sorted(os.listdir(image_dir)):
            image_class_dir = os.path.join(image_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)
            if not os.path.isdir(image_class_dir) or not os.path.isdir(mask_class_dir):
                continue
            self.class_names.append(class_name)
            class_index = self.class_names.index(class_name)

            for image_name in os.listdir(image_class_dir):
                if image_name.endswith('.jpeg') or image_name.endswith('.JPEG'):  # Adjust as needed
                    image_path = os.path.join(image_class_dir, image_name)
                    mask_name = os.path.splitext(image_name)[0] + '.npy'  # Change to .npy
                    mask_path = os.path.join(mask_class_dir, mask_name)
                    if os.path.exists(mask_path):  # Check if corresponding mask exists
                        self.samples.append((image_path, mask_path, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, class_index = self.samples[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = np.load(mask_path)

        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)

        #scale the mask from 0-10 out of 255 to 0-250 out of 255
        mask = mask * 25

        return (image, mask), class_index
    
    def get_file_paths(self, idx):
        image_path, mask_path, class_index = self.samples[idx]
        return image_path, mask_path

#customized transform so image and mask could have the same random crop and flip
class RandomResizedCropAndFlip:
    def __init__(self, input_size, scale=(0.2, 1.0)):
        self.input_size = input_size
        self.scale = scale
        self.resized_crop = transforms.RandomResizedCrop(size=input_size, scale=scale, interpolation=3)  # Bicubic
        self.flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Resize((input_size, input_size))  # Resize transformation


    def __call__(self, image, mask):
        mask =  Image.fromarray(mask).convert('RGB')

        # Apply the same random crop and flip to both image and mask
        i, j, h, w = self.resized_crop.get_params(image, self.scale, self.resized_crop.ratio)
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Resize both image and mask
        image = self.resize(image)
        mask = self.resize(mask)

        image = self.to_tensor(image)
        # image = self.normalize(image)
        mask = self.to_tensor(mask)
        # add the mask one more dimension as channel
        # mask = mask.unsqueeze(0)

        return image, mask
    
    