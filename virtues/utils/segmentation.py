import torch
from torch import nn
from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch_mean
import torch.nn.functional as F
import numpy as np

class SegmentationTilePredictor(nn.Module):
    def __init__(self, virtues_model: nn.Module, channels: torch.Tensor):
        super().__init__()
        self.virtues_model = virtues_model
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = [img for img in x]
        channels = [self.channels.to(x.device) for _ in range(x.shape[0])]
        return self.virtues_model(images, channels)

def tiled_logits_inference(segmentation_model: nn.Module, tissue_image: torch.Tensor, channels: torch.Tensor, tile: int, ovlp: int, bs: int, device="cuda"):
    h, w = int(tissue_image.shape[-2]), int(tissue_image.shape[-1])
    tile_hw = (min(tile, h), min(tile, w))
    chop_idx = _chops(tissue_image.shape, shape=tile_hw, overlap=2 * ovlp)
    tiles = _tiles_from_chops(tissue_image, shape=tile_hw, tuple_index=chop_idx)

    logits_tiles = []
    use_amp = isinstance(device, torch.device) and device.type == "cuda"

    with torch.no_grad():
        for i in range(0, len(tiles), bs):
            image_batch = torch.stack(tiles[i : i + bs]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = segmentation_model(image_batch, [channels] * len(image_batch))

            pred = pred.detach()
            if pred.shape[-2:] != tile_hw:
                pred = F.interpolate(pred, size=tile_hw, mode="bilinear", align_corners=False)
            logits_tiles.extend([p.cpu() for p in pred])

    stitched_logits = _stitch_mean(
        logits_tiles,
        shape=tile_hw,
        chop_list=chop_idx,
        final_shape=(logits_tiles[0].shape[0], h, w),
    )
    return stitched_logits

def remove_small_cells(prediction: torch.Tensor | np.ndarray, min_cell_size: int = 15) -> np.ndarray:
    labels, counts = np.unique(prediction, return_counts=True)
    small_cells = labels[counts < min_cell_size]
    prediction = np.where(np.isin(prediction, small_cells), 0, prediction)
    labels, inverse = np.unique(prediction, return_inverse=True)
    relabelled = inverse.reshape(prediction.shape)
    return relabelled.astype(np.int32)

def assign_cell_types(instance_prediction: np.ndarray, semantic_prediction: np.ndarray) -> np.ndarray:
    cell_types = np.zeros_like(instance_prediction, dtype=np.int32)
    for cell_id in np.unique(instance_prediction):
        if cell_id == 0:
            continue
        mask = instance_prediction == cell_id
        if np.sum(mask) == 0:
            continue
        cell_type = np.bincount(semantic_prediction[mask]).argmax()
        cell_types[mask] = cell_type
    return cell_types