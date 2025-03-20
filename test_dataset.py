import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, configs, names, transform=None):
        self.config = configs
        self.img_ids = names
        self.img_dir = configs.img_path
        # self.num_classes = 1
        self.transform = transform
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        basename = os.path.basename(img_id)
        img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
        img = np.array(img)
        # img_size = img.shape

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32')
        img = np.transpose(img,(2,0,1))

        return img, {'img_id': basename}

