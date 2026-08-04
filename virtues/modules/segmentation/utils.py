from __future__ import annotations
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.tiling import _chops, _stitch, _stitch_mean, _tiles_from_chops


def segment_large_tissue(
    image: torch.Tensor,
    segmentation_model: nn.Module,
    channel_ids: torch.Tensor,
    tile: int,
    ovlp: int,
    bs: int,
    device: str = "cuda",
    *,
    max_seeds: int = 10000,
    window_size: int = 64,
    detection_size: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict a full-tissue instance mask and semantic logits for tissues too large to
    segment in one shot.

    Args: 
        image: Input image of shape (C, H, W).
        segmentation_model: A segmentation model.
        channel_ids: A tensor of shape (C,) containing the channel IDs for the input image.
        tile: The size of the tiles to use for segmentation.
        ovlp: The amount of overlap between tiles.
        bs: The batch size to use for segmentation.
        device: The device to use for segmentation. Defaults to "cuda".
        max_seeds: The maximum number of seeds to use for instance segmentation. Defaults to 10000.
        window_size: The window size to use for instance segmentation. Defaults to 64.
        detection_size: The detection size to use for instance segmentation. Defaults to 20.
    Returns:
        A tuple containing the predicted instance mask and semantic logits.
    """
    h, w = int(image.shape[-2]), int(image.shape[-1])
    tile_hw = (min(tile, h), min(tile, w))
    chop_idx = _chops(image.shape, shape=tile_hw, overlap=2 * (ovlp + detection_size))
    tiles = _tiles_from_chops(image, shape=tile_hw, tuple_index=chop_idx)

    instance_processor = segmentation_model.instance_processor
    n_instance_channels = int(segmentation_model.dim_out)

    instance_label_tiles = []
    semantic_logit_tiles = []

    with torch.no_grad():
        for i in tqdm(range(0, len(tiles), bs)):
            batch = torch.stack(tiles[i : i + bs]).to(device)
            channels = [channel_ids.to(device)] * len(batch)
            logits = segmentation_model([img for img in batch], channels).detach()
            if logits.shape[-2:] != tile_hw:
                logits = F.interpolate(logits, size=tile_hw, mode="bilinear", align_corners=False)

            for tile_logits in logits:
                instance_label = instance_processor.postprocessing(
                    tile_logits[:n_instance_channels],
                    max_seeds=max_seeds,
                    window_size=window_size,
                    cleanup_fragments=True,
                )
                instance_label_tiles.append(instance_label.cpu())
                semantic_logit_tiles.append(tile_logits[n_instance_channels:].cpu())

    pred_instance, _ = _stitch(
        instance_label_tiles, shape=tile_hw, chop_list=chop_idx, offset=ovlp, final_shape=(1, h, w)
    )
    semantic_logits = _stitch_mean(
        semantic_logit_tiles,
        shape=tile_hw,
        chop_list=chop_idx,
        final_shape=(semantic_logit_tiles[0].shape[0], h, w),
    )
    return pred_instance[0], semantic_logits