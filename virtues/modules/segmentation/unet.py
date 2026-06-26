from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanceProcessor
from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch_mean


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VirtuesTokenUNetDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.embed_dim = embed_dim
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        self.decoder0 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 64),
            # Deconv2DBlock(128, 64),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 128),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, 256),
        )
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.bottleneck_dim))
        self.decoder = self._create_upsampling_branch(out_channels)

    def _create_upsampling_branch(self, num_classes: int) -> nn.Module:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256),
            Conv2DBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128),
            Conv2DBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, padding=0),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64),
            Conv2DBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
        )
        return nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

    def _tokens_to_feature_map(self, z: torch.Tensor) -> torch.Tensor:
        patch_dim = int(np.sqrt(z.shape[-2]))
        return z.transpose(-1, -2).contiguous().view(-1, self.embed_dim, patch_dim, patch_dim)

    def forward(self, z0: torch.Tensor, intermediate_layers: torch.Tensor) -> torch.Tensor:
        z0 = self._tokens_to_feature_map(z0)
        z1 = self._tokens_to_feature_map(intermediate_layers[0])
        z2 = self._tokens_to_feature_map(intermediate_layers[1])
        z3 = self._tokens_to_feature_map(intermediate_layers[2])
        z4 = self._tokens_to_feature_map(intermediate_layers[3])
        return self._forward_upsample(z0, z1, z2, z3, z4, self.decoder)


class VirtuesSegmentationHead(nn.Module):
    def __init__(self, virtues_model, dim_out: int, num_celltypes=None, intermediate_layers=[4, 8, 12, 16]):
        super().__init__()
        self.virtues_model = virtues_model
        self.dim_out = dim_out
        model_dim = int(self.virtues_model.encoder.patch_summary_token.shape[0])
        self.decoder = VirtuesTokenUNetDecoder(embed_dim=model_dim, out_channels=dim_out)
        self.intermediate_layers = intermediate_layers

        self.num_celltypes = num_celltypes
        if num_celltypes is not None:
            self.decoder_phenotypes = VirtuesTokenUNetDecoder(embed_dim=model_dim, out_channels=num_celltypes)

        self.instance_processor = InstanceProcessor(
            device="cuda",
            n_sigma=2,
            cells_and_nuclei=False,
            window_size=64,
        )
        self.instance_processor.initialize_pixel_classifier(self, MLP_width=5)
        

    def _encode(self, mx_images, channels):
        amp_enabled = mx_images[0].device.type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                virtues_output = self.virtues_model.encoder.forward_list(mx_images, channels, return_intermediate_layers=self.intermediate_layers)
        
        ps = virtues_output.patch_summary_tokens
        intermediate_layers = torch.stack(list(virtues_output.intermediate_representations.values()))

        ps = torch.stack(ps, dim=0)
        z0 = rearrange(ps, "b h w d -> b (h w) d")
        return z0, intermediate_layers

    def forward(self, mx_images, channels):

        z0, intermediate_layers = self._encode(mx_images, channels)
        out = self.decoder(z0, intermediate_layers)
        if self.num_celltypes is not None:
            phenotypes = self.decoder_phenotypes(z0, intermediate_layers)
            out = torch.cat([out, phenotypes], dim=1)

        return out

    @torch.no_grad()
    def segment_tile(self, multiplex, channel_ids):
        """
        Computes cell segmentation and instance segmentation logits for the given multiplexed image and channel ids.
        Args:
            multiplex (torch.Tensor):A single multiplexed image to segment. The image should be of shape (C,H,W).
            channel_ids (torch.Tensor): Channel ids corresponding to the channels in the multiplexed images. The tensor should be of shape (C,).
        Returns:
            pred_instance (torch.Tensor): The predicted instance segmentation mask. The tensor will be of shape (H,W) with integer values representing different instances.
            semantic_logits (torch.Tensor): The predicted semantic segmentation logits. The tensor will be of shape (num_classes,H,W).
        """
        logits = self.forward([multiplex], [channel_ids])[0]
        inst_logits = logits[: self.dim_out]
        pred_instance = self.instance_processor.postprocessing(inst_logits, window_size=64, cleanup_fragments=True)[0]
        semantic_logits = logits[self.dim_out:, :, :]
        return pred_instance, semantic_logits
    
    @torch.no_grad()
    def segment_tissue(self,
                    multiplex_tissue,
                    channel_ids,
                    tile_size,
                    overlap,
                    batch_size,
                    ):
        """
        Computes cell segmentation and instance segmentation logits for a large multiplexed tissue image by processing it in tiles.
        """
        h, w = int(multiplex_tissue.shape[-2]), int(multiplex_tissue.shape[-1])
        tile_hw = (min(tile_size, h), min(tile_size, w))
        chop_idx = _chops(multiplex_tissue.shape, shape=tile_hw, overlap=2 * overlap)
        tiles = _tiles_from_chops(multiplex_tissue, shape=tile_hw, tuple_index=chop_idx)

        logits_tiles = []

        for i in range(0, len(tiles), batch_size):
            image_batch = torch.stack(tiles[i : i + batch_size])
            # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            pred = self.forward(image_batch, [channel_ids] * len(image_batch))

            pred = pred.detach()
            if pred.shape[-2:] != tile_hw:
                pred = F.interpolate(pred, size=tile_hw, mode="bilinear", align_corners=False)
            logits_tiles.extend([p for p in pred])

        stitched_logits = _stitch_mean(
            logits_tiles,
            shape=tile_hw,
            chop_list=chop_idx,
            final_shape=(logits_tiles[0].shape[0], h, w),
        )

        inst_logits = stitched_logits[: self.dim_out]
        pred_instance = self.instance_processor.postprocessing(inst_logits, window_size=64, cleanup_fragments=True)[0]
        semantic_logits = stitched_logits[self.dim_out:, :, :]
        return pred_instance, semantic_logits