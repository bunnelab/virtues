import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets.augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from utils.utils import load_marker_embedding_dict
from utils.masking import generate_mask
from typing import Tuple
import math

class MultiplexDataset(Dataset):

    def __init__(
            self,
            tissue_dir: str,
            crop_dir : str,
            mask_dir: str,
            tissue_index: str,
            crop_index : str,
            channels_file : str,
            quantiles_file : str,
            means_file : str,
            stds_file : str,
            marker_embedding_dir : str,
            split : str = 'all',
            crop_size : int = 128,
            patch_size : int = 8,
            masking_ratio : Tuple[float, float] = (0.6, 1.0),
            channel_fraction : Tuple[float, float] = (0.75, 1.0),
        ):
        """
        tissue_dir: directory containing full multiplex images
        crop_dir: directory containing cropped multiplex images for training
        mask_dir: directory containing segmentation masks
        crop_index: path to csv file containing crop metadata
        channels_file: path to csv file containing channel information in the same order as the multiplex images
        quantiles_file: path to csv file containing per-channel 99th percentile values for clipping
        means_file: path to csv file containing per-channel mean values for normalization
        stds_file: path to csv file containing per-channel std values for normalization
        split: split to use (train, test, all)
        crop_size: size of the training crops
        patch_size: size of the patches for masking
        masking_ratio: tuple indicating the range of masking ratios from which per-sample masking ratio is drawn uniformly
        channel_fraction: tuple indicating the range of channel fractions from which per-sample fraction for channel dropout is drawn uniformly
        """
        
        self.tissue_dir = tissue_dir
        self.crop_dir = crop_dir
        self.mask_dir = mask_dir

        self.tissue_index = pd.read_csv(tissue_index)
        self.crop_index = pd.read_csv(crop_index)

        self.channels = pd.read_csv(channels_file)

        self.quantiles = pd.read_csv(quantiles_file, index_col=0)
        self.means = pd.read_csv(means_file, index_col=0)
        self.stds = pd.read_csv(stds_file, index_col=0)

        self.channel_mask = [] 
        self.marker_indices = [] 
        marker_embedding_dict = load_marker_embedding_dict(marker_embedding_dir)
        for i, row in self.channels.iterrows():
            protein_id = row['protein_id']
            if protein_id in marker_embedding_dict:
                self.channel_mask.append(True)
                self.marker_indices.append(marker_embedding_dict[protein_id])
            else:
                self.channel_mask.append(False)
        self.channel_mask = torch.tensor(self.channel_mask, dtype=torch.bool)
        self.marker_indices = torch.tensor(self.marker_indices, dtype=torch.long)

        self.split = split

        if split != 'all':
            self.tissue_index = self.tissue_index.query(f'split == "{split}"')
            self.crop_index = self.crop_index[self.crop_index['tissue_id'].isin(self.tissue_index['tissue_id'])]

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio

        self.random_crop = MultiplexRandomCrop(size=(crop_size, crop_size))
        self.random_symmetry = MultiplexRandomSymmetry()
        self.gaussian_blur = v2.GaussianBlur(kernel_size=3, sigma=(1.0))
        self.drop_channels = ChannelDropout(channel_fraction=channel_fraction)


    def __len__(self):
        return len(self.crop_index)

    def __getitem__(self, idx: int):
        row = self.crop_index.iloc[idx]
        tissue_id = row['tissue_id']
        crop_id = row['crop_id']
        
        multiplex = self.get_crop(tissue_id, crop_id, preprocess=True)

        marker_indices = self.marker_indices

        multiplex, marker_indices = self._augment(multiplex, marker_indices)
    
        C = multiplex.shape[0]
        H = W = self.crop_size // self.patch_size
        mask = generate_mask(C, H, W, self.masking_ratio)

        return multiplex, marker_indices, mask
    
    def get_tissue(self, tissue_id: str, preprocess: bool = True):
        """
        Returns the full multiplex image for the given tissue_id including per-default preprocessing.
        """
        path = f"{self.tissue_dir}/{tissue_id}.npy"
        multiplex = np.load(path)
        multiplex = torch.tensor(multiplex).float()
        multiplex = multiplex[self.channel_mask]
        if preprocess:
            multiplex = self._preprocess(tissue_id, multiplex)
        return multiplex
    
    def get_crop(self, tissue_id: str, crop_id: int, preprocess: bool = True):
        """
        Returns the specified crop for the given tissue_id including per-default preprocessing.
        """
        path = f"{self.crop_dir}/{tissue_id}_{crop_id}.npy"
        multiplex = np.load(path)
        multiplex = torch.tensor(multiplex).float()
        multiplex = multiplex[self.channel_mask]
        if preprocess:
            multiplex = self._preprocess(tissue_id, multiplex)
        return multiplex
    
    def get_segmentation_mask(self, tissue_id: str):
        """
        Returns the segmentation mask for the given tissue_id.
        """
        path = f"{self.mask_dir}/{tissue_id}.npy"
        seg_mask = np.load(path)
        seg_mask = torch.tensor(seg_mask).long()
        return seg_mask

    def get_marker_indices(self):
        """
        Returns the marker indices corresponding to the channels used in the dataset.
        """
        return torch.tensor(self.marker_indices, dtype=torch.long)
    
    def _preprocess(self, tissue_id: str, multiplex: torch.Tensor):
        """
        Applies image-preprocessing steps.
        """
        # 1. Clipping at 99th tissue percentile
        quantiles = self.quantiles.loc[tissue_id].values[self.channel_mask]
        quantiles = torch.from_numpy(quantiles).float()[:, None, None]
        min_ = torch.zeros_like(quantiles)
        multiplex = torch.clamp(multiplex, min=min_, max=quantiles)
        
        # 2. Log1p normalization
        multiplex = torch.log1p(multiplex)

        # 3. Gaussian Blur
        multiplex = self.gaussian_blur(multiplex)

        # 4. Log-Standardization
        log_mean = self.means.loc[tissue_id].values[self.channel_mask]
        log_std = self.stds.loc[tissue_id].values[self.channel_mask]
        log_mean = torch.from_numpy(log_mean).float()[:, None, None]
        log_std = torch.from_numpy(log_std).float()[:, None, None]

        multiplex = (multiplex - log_mean) / (log_std + 1e-9)
        return multiplex

    def _augment(self, multiplex : torch.Tensor, marker_indices : torch.Tensor):
        """
        Applies image augmentations.
        multiplex: Tensor of shape (C, H, W)
        marker_indices: Tensor of shape (C,)
        """
        # 1. Random crop
        multiplex = self.random_crop(multiplex)

        # 2. Random symmetry
        multiplex = self.random_symmetry(multiplex)

        # 3. Random channel dropout
        multiplex, marker_indices = self.drop_channels(multiplex, marker_indices)
        return multiplex, marker_indices
    

        


