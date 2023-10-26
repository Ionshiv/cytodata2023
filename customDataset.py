from glob import glob
import os
import torch as tch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path


CROP_SIZE = 1024  # Adjust the crop size as needed
TRANSLATION_PARAMETER = 512  # Adjust the translation parameter as needed

def quantile_clip(image_array, q_min=0.01, q_max=0.99):
    # Ensure that the input image is 2D or 3D NumPy array
    if len(image_array.shape) not in [2, 3]:
        raise ValueError("Input image should be a 2D or 3D NumPy array.")

    # Calculate the quantiles
    min_quantile = np.percentile(image_array, 100 * q_min)
    max_quantile = np.percentile(image_array, 100 * q_max)

    # Scale the image based on quantiles
    scaled_image = np.clip(image_array, min_quantile, max_quantile)

    return scaled_image


def second_lowest_intensity(image):
    # Find the minimum and maximum values in the image
    min_value = np.min(image)
    max_value = np.max(image)

    # Replace the minimum value with the maximum value to find the second lowest
    image_with_max = np.where(image == min_value, max_value, image)

    # Find the minimum value in the modified image
    second_lowest = np.min(image_with_max)

    return second_lowest


def log_scale_and_normalize(image, quantile=True):
    # Apply quantile clipping
    clipped_image = quantile_clip(image)

    # Apply the logarithmic scaling
    log_scaled_image = np.log1p(clipped_image)  # Add 1 to avoid taking the log of zero

    # Min-Max normalization to [0, 1]
    min_value = second_lowest_intensity(log_scaled_image)
    max_value = np.max(log_scaled_image)
    normalized_image = (log_scaled_image - min_value) / (max_value - min_value)
    normalized_image = np.clip(normalized_image, 0, 1)

    return normalized_image


def generate_cropped_images(image, crop_size, translation_parameter):
    # Get the dimensions of the input image
    height, width = image.shape[0], image.shape[1]

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Initialize an array to store the cropped images
    cropped_images = []

    # Define translation offsets for up, down, left, and right
    translations = [(0, -translation_parameter), (0, translation_parameter),
                    (-translation_parameter, 0), (translation_parameter, 0)]

    for translation in translations:
        # Calculate the crop coordinates
        crop_x1 = max(center_x - crop_size // 2 + translation[0], 0)
        crop_x2 = min(center_x + crop_size // 2 + translation[0], width)
        crop_y1 = max(center_y - crop_size // 2 + translation[1], 0)
        crop_y2 = min(center_y + crop_size // 2 + translation[1], height)

        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Append the cropped image to the list
        cropped_images.append(cropped)

    # Stack the cropped images into an array of shape [B, N, M]
    cropped_images = np.stack(cropped_images)

    return cropped_images



class customDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Define the root path and img_path (you need to define img_path first)
        meta_path = os.path.join(self.root_dir, 'metadata')
        img_path = os.path.join(self.root_dir, 'images')
        trainmeta_path = os.path.join(meta_path, 'cytodata2023_hackathon_train.csv')
        df_metadata_train = pd.read_csv(trainmeta_path)

        # Create an empty list to store the paths
        self.all_files = []

        # Iterate through the rows in df_metadata_train
        for idx, row in df_metadata_train.iterrows():
            slide_pattern = row['Slide'] + '*roi{:03d}*'.format(row['ROI number'])
            imgpaths = list(Path(img_path).glob(slide_pattern))
            imgpaths.sort()
            self.all_files.append(imgpaths)

        self.channels = [f"c{i}" for i in range(1, 7)]  # c1 to c6

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        images = []
        for j, _ in enumerate(self.channels):
            img_name = self.all_files[idx][j]
            image = Image.open(img_name)
            # [N, M]
            image = log_scale_and_normalize(image, quantile=True)
            image = generate_cropped_images(image, CROP_SIZE, TRANSLATION_PARAMETER)
            # [4, N, M]
            image = transforms.ToTensor()(image)
            images.append(image)

        # Stack along new dimension to create a single tensor for the multi-channel image
        multi_channel_img = tch.stack(images)  # [C, 4, N, M]
        multi_channel_img = multi_channel_img.permute(1, 0, 2, 3)  # [4, C, N, M]
        multi_channel_img = tch.squeeze(multi_channel_img, 2)
        
        return multi_channel_img.to(tch.float32)

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