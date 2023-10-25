from glob import glob
import os
import torch as tch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_files = glob(os.path.join(root_dir, 'unit*_c*.png'))
        
        # Extract unique unit IDs
        self.unit_ids = list(set(f.split('_')[0] for f in [os.path.basename(x) for x in self.all_files]))
        self.unit_ids.sort()  # sort them if needed
        
        self.channels = [f"c{i}" for i in range(1, 7)]  # c1 to c6

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, idx):
        unit_id = self.unit_ids[idx]
        images = []
        for channel in self.channels:
            img_name = os.path.join(self.root_dir, f"{unit_id}_{channel}.png")
            image = Image.open(img_name)
            image = transforms.ToTensor()(image)
            images.append(image)
        
        # Stack along new dimension to create a single tensor for the multi-channel image
        multi_channel_img = tch.stack(images)
        
        return multi_channel_img

# from torch.utils.data import Dataset
# from torchvision import transforms
# import PIL.Image as Image
# import os
# import torch as tch

# class customDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.unit_ids = [f"unit{str(i).zfill(2)}" for i in range(1, 21)]  # unit01 to unit50
#         self.channels = [f"c{i}" for i in range(1, 7)]  # c1 to c6

#     def __len__(self):
#         return len(self.unit_ids)

#     def __getitem__(self, idx):
#         unit_id = self.unit_ids[idx]
#         images = []
#         for channel in self.channels:
#             img_name = os.path.join(self.root_dir, f"{unit_id}_{channel}.png")
#             image = Image.open(img_name)
#             image = transforms.ToTensor()(image)
#             images.append(image)
        
#         # Stack along new dimension to create a single tensor for the multi-channel image
#         multi_channel_img = tch.stack(images)
        
#         return multi_channel_img

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# import torch as tch
# from sklearn.model_selection import KFold
# from PIL import Image

# class customDataset(tch.utils.data.Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.unit_ids = sorted(set(f.split('_')[0] for f in os.listdir(img_dir)))

#     def __len__(self):
#         return len(self.unit_ids)

#     def __getitem__(self, idx):
#         unit_id = self.unit_ids[idx]
#         channels = []
#         for c in range(1, 7):
#             print(f'unit{unit_id}_c{c}')
#             img_path = os.path.join(self.img_dir, f"{unit_id}_c{c}.png")
#             img = Image.open(img_path).convert("L")
#             if self.transform:
#                 img = self.transform(img)
#             channels.append(img)
#         x = tch.stack(channels, dim=0)
#         print(x.shape)
#         return x