import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from virtues.data.augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from virtues.utils.utils import load_marker_embedding_dict
from virtues.utils.masking import generate_mask
from typing import Tuple
import math
from spora_io import MultiplexImagingDataset


class VirTuesPretrainingDataset(Dataset):

    def __init__(self,
            multiplex_dataset: MultiplexImagingDataset,
            marker_embeddings_dir: str,
            split: str = 'train',
            patch_size: int = 8,
            masking_ratio : Tuple[float, float] = (0.6, 1.0),
            channel_fraction : Tuple[float, float] = (0.75, 1.0),
        ):
        """
        multiplex_dataset: MultiplexImagingDataset
        marker_embeddings_dir: path to directory containing marker embedding files
        split: split to use, "all" uses all data
        patch_size: size of the patches for masking
        masking_ratio: tuple indicating the range of masking ratios from which per-sample masking ratio is drawn uniformly
        channel_fraction: tuple indicating the range of channel fractions from which per-sample fraction for channel dropout is drawn uniformly
        """
        self.multiplex_dataset = multiplex_dataset
        self.marker_embedding_dict = load_marker_embedding_dict(marker_embeddings_dir) # maps uniprot id to embedding index
        self.split = split
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.channel_fraction = channel_fraction

        self.tiles = []
        tissue_metadata = self.multiplex_dataset.tissue_modality_metadata
        if split != 'all':
            tissue_metadata = tissue_metadata[tissue_metadata['split'] == split]
        tissue_ids = tissue_metadata.index.values
        for tissue_id in tissue_ids:
            tile_coords = self.multiplex_dataset.tile_coordinates.get(tissue_id, []) # list of (row, col) tuples
            for row, col in tile_coords:
                self.tiles.append(({'tissue_id': tissue_id, 'row': row, 'col': col}))
        self.tiles = pd.DataFrame(self.tiles)

        self.random_symmetry = MultiplexRandomSymmetry()
        self.drop_channels = ChannelDropout(channel_fraction=channel_fraction)

    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx: int):
        # Load the full multiplex image for the tile
        tile = self.tiles.iloc[idx]
        tissue_id = tile['tissue_id']
        row, col = tile['row'], tile['col']
        tissue = self.multiplex_dataset.get_tile_by_coordinates(tissue_id=tissue_id, row=row, col=col, preprocess=True, kind='uniprot_filtered')
        img = tissue.image
        uniprots = tissue.uniprot_ids
        marker_indices = torch.tensor([self.marker_embedding_dict[uniprot] for uniprot in uniprots], dtype=torch.long)

        # Apply augmentations
        img, marker_indices = self._augment(img, marker_indices)

        # Generate mask
        C = img.shape[0]
        H = img.shape[1] // self.patch_size
        W = img.shape[2] // self.patch_size
        mask = generate_mask(C, H, W, self.masking_ratio)
        return img, marker_indices, mask

    def _augment(self, multiplex : torch.Tensor, marker_indices : torch.Tensor):
        """
        Applies image augmentations.
        multiplex: Tensor of shape (C, H, W)
        marker_indices: Tensor of shape (C,)
        """
        # 1. Random symmetry
        multiplex = self.random_symmetry(multiplex)
        # 2. Random channel dropout
        multiplex, marker_indices = self.drop_channels(multiplex, marker_indices)
        return multiplex, marker_indices