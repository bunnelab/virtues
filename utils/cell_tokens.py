import numpy as np
import torch

from modules.multiplex_virtues import MultiplexVirtues


def _get_uniform_crops(img, stride, crop_size=128):
    crops = []
  
    h, w = img.shape[-2:]
    indices_set = set()
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            indices_set.add((i, j))
    
    last_i = h - crop_size
    for j in range(0, w - crop_size + 1, stride):
        indices_set.add((last_i, j))
    last_j = w - crop_size
    for i in range(0, h - crop_size + 1, stride):
        indices_set.add((i, last_j))
    indices_set.add((last_i, last_j))
    indices = sorted(list(indices_set))
    for i, j in indices:
        crop = img[:, i:i+crop_size, j:j+crop_size]
        crops.append(crop)
    # turn the indices back to the original coordinates
    for i in range(len(indices)):
        indices[i] = (indices[i][0], indices[i][1])
    return crops, indices

def _assign_patch_tokens_to_cells(crop_tokens, crop_mask, patch_size):
    cell_tokens = {} 
    weights = {}  
    for i in range(crop_tokens.shape[0]): 
        for j in range(crop_tokens.shape[1]):
            patch_mask = crop_mask[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            unique, counts = np.unique(patch_mask, return_counts=True)
            patch_cell_coverage = dict(zip(unique, counts))
            for cell_id, overlap_pixels in patch_cell_coverage.items():
                if cell_id == 0:  
                    continue
                if cell_id not in weights:
                    weights[cell_id] = []
                weights[cell_id].append(overlap_pixels)
                if cell_id not in cell_tokens:
                    cell_tokens[cell_id] = []
                cell_tokens[cell_id].append(crop_tokens[i, j, :])
    return cell_tokens, weights

def compute_cell_tokens(model : MultiplexVirtues, 
                        img : torch.Tensor, 
                        channel : torch.Tensor, 
                        segmentation_mask: torch.Tensor, 
                        device='cuda',
                        crop_size=128, 
                        patch_size=8, 
                        stride=42, 
                        chunk_size=32) -> torch.Tensor: 
    '''
    Compute cell tokens
    Args:
        model: MultiplexVirtues model
        img: input image tensor of shape (C, H, W)
        channel: channel tensor of shape (C,)
        segmentation_mask: cell segmentation mask of shape (H, W)
        device: device to run the model on
        crop_size: size of the crops to extract
        patch_size: size of the patches in the model
        stride: stride for extracting crops
        chunk_size: number of crops to process in a chunk
    Returns:
        cell_ids: list of cell ids
        cell_tokens: tensor of shape (num_cells, token_dim)
    '''
    crops, indices = _get_uniform_crops(img, stride, crop_size=crop_size)
    crops = [crops.to(device) for crops in crops]
    channel = channel.to(device)
        
    crop_tokens = []
    for i in range(0, len(crops), chunk_size):
        crops_chunk = crops[i:i+chunk_size]
        channels_chunk = [channel for _ in range(len(crops_chunk))]
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                encoder_output = model.encoder.forward_list(
                    crops_chunk,
                    channels_chunk,
                    multiplex_mask=None,
                )
                pss = encoder_output.patch_summary_tokens
                pss = [ps.cpu() for ps in pss]
                crop_tokens.extend(pss)
                
    cell_tokens = {}
    weights = {}

    for crop_token, (row, col) in zip(crop_tokens, indices):
        crop_mask = segmentation_mask[row:row + crop_size, col:col + crop_size]
        crop_cell_tokens, crop_weights = _assign_patch_tokens_to_cells(crop_token, crop_mask, patch_size)
        for cell_id, tokens in crop_cell_tokens.items():
            if cell_id not in cell_tokens:
                cell_tokens[cell_id] = []
            if cell_id not in weights:
                weights[cell_id] = []
            cell_tokens[cell_id].extend(tokens)
            weights[cell_id].extend(crop_weights[cell_id])
    
    avg_cell_tokens = {}
    for cell_id, tokens in cell_tokens.items():
        
        tokens = torch.stack(tokens)
        weights_array = torch.tensor(weights[cell_id], dtype=tokens.dtype, device=tokens.device)
        weights_array = weights_array / weights_array.sum()
        avg_cell_token = torch.sum(tokens * weights_array[:, None], dim=0)
        avg_cell_tokens[cell_id] = avg_cell_token
    cell_ids = list(avg_cell_tokens.keys())
    cell_ids.sort()
    cell_tokens = torch.stack([avg_cell_tokens[cell_id] for cell_id in cell_ids])
    return cell_ids, cell_tokens, crop_tokens, indices