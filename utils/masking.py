import random
import torch
import math
from typing import Tuple

def generate_mask(C : int, H : int, W : int, masking_ratio: Tuple[float, float]) -> torch.Tensor:
        """
        Generates a random multi-channel mask with independent masking for each channel.
        C: number of channels
        H: grid height of the mask
        W: grid width of the mask
        """
        masks = []
        for _ in range(C):
            mask_ratio = random.uniform(masking_ratio[0], masking_ratio[1])
            mask_area = math.ceil(H * W * mask_ratio)
            mask = torch.zeros(H*W, dtype=bool)
            mask[:mask_area] = True
            mask = mask[torch.randperm(H*W)].reshape(H, W)
            masks.append(mask)
        mask = torch.stack(masks, dim=0)
        return mask