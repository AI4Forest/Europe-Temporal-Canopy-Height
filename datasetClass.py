from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import os
import torch
import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys
from typing import Optional
from torchvision.transforms import transforms

import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys
from typing import Optional
from torch.utils.data import Dataset

class SatelliteImageDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path: str,
                 shift_year: int, 
                 collapse_months: bool = False, 
                 single_month_scaling: bool = True, 
                 scale_adjustments: dict = None, 
                 is_3d_model: bool = False,
                 time_mode: Optional[str] = None,
                 has_sentinel_1: bool = False):
        
        self.shift_year = shift_year
        assert collapse_months in [True, False], "collapse_months must be either True or False."
        self.collapse_months = collapse_months
        self.is_3d_model = is_3d_model
        self.single_month_scaling = single_month_scaling
        if self.single_month_scaling and self.collapse_months:
            sys.stderr.write("Warning: Cannot use single_month_scaling = True with collapse_months = True, setting single_month_scaling to False.\n")
            self.single_month_scaling = False
        
        if self.is_3d_model:
            if not self.single_month_scaling:
                sys.stderr.write("Warning: single_month_scaling must be True for 3D models, setting single_month_scaling to True to avoid NaN values.\n")
                self.single_month_scaling = True
        
        # Assert that the time mode is valid
        assert time_mode in ['channel', 'rescale', None, 'None', 'none'], "time_mode must be either 'channel', 'rescale', or None."
        if time_mode in ['None', 'none']:
            self.time_mode = None
        else:
            self.time_mode = time_mode

        self.has_sentinel_1 = has_sentinel_1

        # Print part of the configuration
        sys.stdout.write(f"Single month scaling: {self.single_month_scaling}.\n")
        sys.stdout.write(f"Collapse months: {self.collapse_months}.\n")
        sys.stdout.write(f"Is 3D model: {self.is_3d_model}.\n")
        sys.stdout.write(f"Time mode: {self.time_mode}.\n")

        df = pd.read_csv(os.path.join(data_path, "metadata.csv"))

        # Step 1: Join the 'sentinel_file' column with 'data_path'
        df["files"] = df["tile"].apply(lambda x: os.path.join(data_path, 'samples', x, x))

        # Step 2: Append '_' to the end of the file name
        df["files"] = df["files"] + "_"

        # Step 3: Append the value in column 'sample_id' to the end of the file name
        df["files"] = df["files"] + df["sample_id"].astype(str)

        # Step 4: Add '.npz' to the end of the file name
        df["files"] = df["files"] + ".npz"

        # Assign the result to self.files
        self.files = np.array(df["files"]).astype(np.bytes_)
        self.year_data = df["year"].values.astype(np.float32)
        

        self.scale_adjustments = scale_adjustments or {}
        self.scaling_dict = self.get_adjusted_scaling_dict()

    def get_adjusted_scaling_dict(self):
        base_dict = {
            (1, 2, 3, 4): (0, 2000),
            (6, 7, 8, 9): (0, 6000),
            (0,): (0, 1000),
            (5, 10, 11): (0, 4000),
        }

        adjusted_dict = {}
        for channels, (min_val, max_val) in base_dict.items():
            if channels == (1, 2, 3, 4):
                adjustment = self.scale_adjustments.get('scale_adjust_1234', 0.0)
            elif channels == (6, 7, 8, 9):
                adjustment = self.scale_adjustments.get('scale_adjust_6789', 0.0)
            elif channels == (0,):
                adjustment = self.scale_adjustments.get('scale_adjust_0', 0.0)
            elif channels == (5, 10, 11):
                adjustment = self.scale_adjustments.get('scale_adjust_51011', 0.0)
                
            else:
                adjustment = 0.0
            
            if adjustment is None:
                adjustment = 0.0

            adjusted_max = max_val * (1 + adjustment)
            adjusted_dict[channels] = (min_val, adjusted_max)
        sys.stdout.write(f"Adjusted scaling dict: {adjusted_dict}.\n")

        return adjusted_dict
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)    # Columns: sentinel_data (6x12x256x256), gedi_data (3x256x256, first three channels are heights, slope is not given for now). if has_sentinel_1 is True, then sentinel_data is split as seen below
        if self.has_sentinel_1:
            image_s1 = data["sentinel1_data"].astype(np.float32)    # 4x256x256
            
            # Replace any NaN values with 0
            image_s1[np.isnan(image_s1)] = 0

            image_s2 = data["sentinel2_data"].astype(np.float32)    # 6x12x256x256

            # Sentinel 1 channels are given in range -50 to 1 and should be scaled to 0 to 1
            image_s1 = ((image_s1 + 50) / 51)

            # Create the image by concatenating the first 4 channels of image_s1 and the second 12 channels of image_s2, resulting in a tensor of shape 6x16x256x256. Therefore, the s1 channels must be repeated first, to bring the shape to 6x4x256x256
            image_s1 = np.repeat(image_s1[np.newaxis, ...], repeats=image_s2.shape[0], axis=0)  # 6x4x256x256
            image = np.concatenate((image_s2, image_s1), axis=1)  # 6x16x256x256, concatenate first the s2 channels, then the s1 channels, such that the scaling is correctly applied
        else:
            image = data["sentinel_data"].astype(np.float32)
        if self.collapse_months:
            # Collapse the first dimension by computing the median along this dimension
            original_shape_of_month = image.shape[1:]
            image = image.reshape(image.shape[0], -1)
            image = np.median(image, axis=0)
            # Reshape back to the original shape of the month
            image = image.reshape(*original_shape_of_month)
        else:
            # Potentially apply single month scaling
            if self.single_month_scaling:
                for channels, (min_val, max_val) in self.scaling_dict.items():
                    for channel in channels:
                        image[:, channel] = np.clip(image[:, channel], min_val, max_val)
                        image[:, channel] = (image[:, channel] - min_val) / (max_val - min_val)
            if not self.is_3d_model:
                # Just join the first two dimensions into one, e.g. the resulting shape is either (6*12)x256x256 or (12*12)x256x256 (or use 16 instead of 12 if has_sentinel_1 is True)
                image = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
            else:
                # Swap month and channel dimensions: (months, channels, H, W) -> (channels, months, H, W)
                image = image.transpose(1, 0, 2, 3) # This is equivalent to image.permute(1, 0, 2, 3) and should also create a view instead of a copy
        
        # Transform the image into a torch tensor without swapping dimensions
        # Remark: Suboptimal solution. It works and avoids the dimension swapping of transforms.ToTensor(), but it also does not meet the complexity of ToTensor()
        image = torch.from_numpy(image).contiguous()

        # Concatenate the year variable as an additional channel if time_mode is 'channel'
        year = self.year_data[index] - self.shift_year
        year = np.array([year], dtype=np.float32)  # Ensure the correct shape
        if self.time_mode == 'channel':            
            if self.is_3d_model:
                year_channel = np.full((1, image.shape[-3], image.shape[-2], image.shape[-1]), year)
            else:
                year_channel = np.full((1, image.shape[-2], image.shape[-1]), year)
            
            image = np.concatenate((image, year_channel), axis=0)   

        year = torch.tensor(year)
        
        label = data["gedi_data"].astype(np.float32)
        label = torch.from_numpy(label).contiguous()

        return image, label, year