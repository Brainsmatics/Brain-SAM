import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, names, transform=None):
        self.config = args
        self.img_ids = names
        self.img_dir = args.img_path
        self.mask_dir = args.label_path
        self.num_classes = 1
        self.transform = transform
        self.format_img = args.format_img
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id+self.format_img)).convert('RGB')
        img = np.array(img)
        mask = []
        for i in range(self.num_classes):
            temp_mask = np.array(Image.open(os.path.join(self.mask_dir, img_id+self.format_img)).convert('L'))

            mask.append(temp_mask)
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32')
        img = np.transpose(img,(2,0,1))
        # mask = mask/255

        mask = np.transpose(mask,(2,0,1))

        # return img, mask, {'img_id': img_id}
        return img, mask
