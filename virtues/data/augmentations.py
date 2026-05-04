import torch
import random
from typing import Tuple
import numpy as np
import math

class MultiplexRandomCrop():

    def __init__(self, size : Tuple[float, float]):
        self.size = size
        
    def __call__(self, image : torch.Tensor):
        """
        image: Tensor of shape (C, H, W)
        """
        _, H, W = image.shape

        target_h = self.size[0]
        target_w = self.size[1]
        
        r = random.randint(0, H - target_h)
        c = random.randint(0, W - target_w)
        
        return image[:, r:r+target_h, c:c+target_w]
        

class RandomRotation(object):

    def __init__(self):
        pass

    def __call__(self, image: torch.Tensor):
        """
        image: Tensor of shape (C, H, W)
        """
        r = random.randint(0, 3)
        if r == 0:
            return image
        elif r == 1:
            return torch.flip(image, (-1,)).transpose(-1, -2)
        elif r == 2:
            return torch.flip(image, (-1, -2))
        elif r == 3:
            return torch.flip(image, (-2,)).transpose(-1, -2)

class MultiplexRandomSymmetry():

    def __init__(self):
        self.random_rotation = RandomRotation()

    def __call__(self, image: torch.Tensor):
        """
        image: Tensor of shape (C, H, W)
        """
        image = self.random_rotation(image)
        if random.random() < 0.5:
            image = image.flip(-1)
        return image

class ChannelDropout():

    def __init__(self, channel_fraction: float):
        self.channel_fraction = channel_fraction

    def __call__(self, multiplex : torch.Tensor, marker_indices : torch.Tensor):
        """
        multiplex: Tensor of shape (C, H, W)
        marker_indices: Tensor of shape (C,)
        """
        C = multiplex.shape[0]
        fraction = random.uniform(*self.channel_fraction)
        C_keep = math.ceil(C * fraction)
        indices = np.random.choice(C, C_keep, replace=False)
        multiplex = multiplex[indices]
        marker_indices = marker_indices[indices]
        return multiplex, marker_indices