import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import random
import numpy as np
import kagglehub


class Flickr8kDataset(Dataset):
    def __init__(self,  dataset="adityajn105/flickr8k", crop_size=180):
        self.get_data(dataset=dataset)
        self.crop_size = crop_size
        self.annotations = []
        self.images = os.listdir(self.image_dir)
        self.max_noise=0.1

    def get_data(self, dataset="adityajn105/flickr8k"):
        path = kagglehub.dataset_download(dataset)
        self.path = path
        self.captions_dir = os.path.join(path, 'captions.txt')
        self.image_dir = os.path.join(path, 'Images')

    def __len__(self):
        return len(self.images)
    
    def get_image(self, idx, top=None, left=None):

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB') 
        w, h = image.size
        if w <= self.crop_size or h <= self.crop_size:
            image = image.resize((self.crop_size*2, self.crop_size*2), Image.BICUBIC)
            w, h = image.size

        if top is None or left is None:
            # Random crop image to crop size
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            top = max(top, 0)
            left = max(left, 0)

        bottom = top + self.crop_size
        right = left + self.crop_size
        image = image.crop((left, top, right, bottom))
        image = np.array(image).astype(np.float16)
        image *= 1./255
        image = image.astype(float)

        conditioning, noise_level = self.compute_conditioning()
        W, H, C = image.shape
        noise = np.random.normal(0, noise_level, [W, H, C])
        noisy_image = np.clip(image + noise, 0, 1)
        return image, noisy_image, conditioning, noise_level
    

    def compute_noise_level(self):
        return self.max_noise
    
    def compute_conditioning(self):
        noise = self.compute_noise_level()
        noise_conditioning = noise/self.max_noise
        return [noise_conditioning, 0], noise

    
    def __getitem__(self, idx):
        image, noisy_image, conditioning, noise_level = self.get_image(idx)

        # Convert to tensor
        noisy_image_tensor = ToTensor()(noisy_image).float()
        image_tensor = ToTensor()(image).float()
        conditioning_tensor  = torch.tensor(conditioning).float()
        output = {
                "noisy_image_tensor": noisy_image_tensor,
                "image_tensor": image_tensor,
                "conditioning_tensor": conditioning_tensor,
        }
        return output
    

