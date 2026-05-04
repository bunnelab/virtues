from typing import Optional, Tuple, Sequence, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.attention_flashattention import MHAwithPosEmb
from modules.layers.basic_modules import build_feedforward
from modules.layers.mask_utils_flashattention import (
    build_self_attention_bias,
    build_self_attention_bias_channel_concat,
    get_non_zero_indices,
)


class TransformerEncoder(nn.Module):
    """
    Stacked pre-LN Transformer encoder.

    Args:
        d_model (int): Model width.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Hidden size of FFN.
        dropout (float): Dropout prob used inside attention and FFN.
        activation (str): Activation used in FFN (e.g., "gelu" / "relu").
        bias (bool): Whether to use bias in linear projections / LayerNorm.
        inbuilt_pos_emb (str|None): Passed to MHAwithPosEmb (e.g., "absolute", "rope", ...).
        num_layers (int): Number of encoder layers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        bias: bool = True,
        inbuilt_pos_emb: Optional[str] = "absolute",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                    inbuilt_pos_emb=inbuilt_pos_emb,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,                        # (B, S, d_model)
        src_pos: Optional[torch.Tensor] = None,   # (B, S, 2)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool or additive
        cu_seq_len: Optional[torch.Tensor] = None,            # FlashAttention varlen
        max_seq_len: Optional[int] = None,                    # FlashAttention varlen
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            (B, S, d_model)
        """
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_pos=src_pos,
                src_key_padding_mask=src_key_padding_mask,
                cu_seq_len=cu_seq_len,
                max_seq_len=max_seq_len,
            )
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single pre-LN Transformer encoder block:
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        bias: bool = True,
        inbuilt_pos_emb: Optional[str] = "absolute",
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.activation = activation

        self.multi_head_attention = MHAwithPosEmb(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            inbuilt_pos_emb=inbuilt_pos_emb,
        )
        # Feedforward: d_model -> dim_feedforward -> d_model with activation/dropout
        self.feedforward = build_feedforward(
            in_dim=d_model,
            out_dim=d_model,
            hidden_dims=dim_feedforward,
            activation_fn=activation,
            use_dropout=True,
            dropout_prob=dropout,
        )

        self.layernorm1 = nn.LayerNorm(d_model, bias=bias)
        self.layernorm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(
        self,
        src: torch.Tensor,                        # (B, S, d_model)
        src_pos: Optional[torch.Tensor] = None,   # (B, S, 2)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
        cu_seq_len: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-LN MHA
        x = src
        x_norm = self.layernorm1(x)
        # print shapes of all in next line
        # print(f"[TransformerEncoderBlock] x_norm: {x_norm.shape}, src_pos: {src_pos.shape if src_pos is not None else None}, src_key_padding_mask: {src_key_padding_mask.shape if src_key_padding_mask is not None else None}, cu_seq_len: {cu_seq_len.shape if cu_seq_len is not None else None}, max_seq_len: {max_seq_len}")
        x = x + self.multi_head_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            query_pos=src_pos,
            key_pos=src_pos,
            key_padding_mask=src_key_padding_mask,
            cu_seq_len=cu_seq_len,
            max_seq_len=max_seq_len,
        )

        # Pre-LN FFN
        x_norm = self.layernorm2(x)
        ff_out = self.feedforward(x_norm)

        x = x + ff_out
        return x


class ChannelAttentionEncoderBlock(nn.Module):
    """
    Encoder over concatenated channels per sample (C x S x D), with varlen FlashAttention.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,           # (B, C, S, D)
        pos: torch.Tensor,         # (B, C, S, 2)
        mask: torch.Tensor,        # (B, C, S) True means masked token
        channels_per_sample: Optional[Sequence[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Masked variant using varlen FlashAttention over *unmasked* tokens.

        Args:
            x: (B, C, S, D)
            pos: (B, C, S, 2)
            mask: (B, C, S), True means masked.
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_cc(
        self,
        x: torch.Tensor,           # (C, S, D)
        pos: torch.Tensor,         # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Attention over channels concatenated per sample (no masking).

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list of channel counts per sample.

        Returns:
            x': (C, S, D) 
        """
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)  # (C, S) all unmasked
        return self.forward_cc_masked(x, pos, mask, channels_per_sample)

    def forward_cc_masked(
        self,
        x: torch.Tensor,           # (C, S, D)
        pos: torch.Tensor,         # (C, S, 2)
        mask: torch.Tensor,        # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Varlen FlashAttention over the subset of tokens where `~mask` is True.

        Returns:
            x': (C, S, D) 
        """
        # Select unmasked rows for varlen pass
        mask_indices = get_non_zero_indices("ChannelAttention_cc_Masked_Mask_indices", ~mask)
        x_false = x[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos[mask_indices].unsqueeze(0) # (1, N, 2)

        # Build varlen cumulative seq-lens for queries
        seq_lens, max_seq_len = build_self_attention_bias(
            "ChannelAttention_cc_masked",
            mask,
            use_true_as_query=False,
        )

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            cu_seq_len=seq_lens,
            max_seq_len=max_seq_len,
        )
        x_proc = out  # (1, N, D)
        x[mask_indices] = x_proc[0]
        return x


class MarkerAttentionEncoderBlock(nn.Module):
    """
    Encoder attending across markers for each spatial position (C as sequence).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,            # (B, C, S, D)
        pos: torch.Tensor,          # (B, C, S, 2)
        mask: torch.Tensor,         # (B, C, S) True means masked
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Varlen FlashAttention over unmasked items across channels for each spatial position.
        """
        B, C, S, D = x.shape
        x_flat = rearrange(x, "B C S D -> (B S) C D")
        pos_flat = rearrange(pos, "B C S D -> (B S) C D")
        mask_flat = rearrange(mask, "B C S -> (B S) C")

        mask_indices = get_non_zero_indices("MarkerAttention_masked_Mask_indices", ~mask_flat)
        x_false = x_flat[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_flat[mask_indices].unsqueeze(0) # (1, N, 2)

        # seq-lens across channels per (B, S) entry
        q_seq_lens, max_seq_len = build_self_attention_bias(
            "MarkerAttention_masked",
            mask_flat,
            use_true_as_query=False,
        )

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            cu_seq_len=q_seq_lens,
            max_seq_len=max_seq_len,
        )
        x_proc = out
        x_flat[mask_indices] = x_proc[0]
        x = rearrange(x_flat, "(B S) C D -> B C S D", B=B)
        return x

    def forward_cc(
        self,
        x: torch.Tensor,               # (C, S, D)
        pos: torch.Tensor,             # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Concatenate sequences by sample (length = channels_i * S) and run varlen attention.

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list of channel counts per sample.
        """
        S = x.shape[1]
        # Per-sample lengths = channels_i * S
        q_lens = torch.as_tensor(channels_per_sample * int(S), device=x.device, dtype=torch.int32) 

        # Flatten to a single batch
        x_pack = rearrange(x, "C S D -> (S C) D").unsqueeze(0)    # (1, total, D)
        pos_pack = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)

        # FlashAttention varlen seq-lens (prepend 0, then cumsum)
        cu = torch.zeros(q_lens.numel() + 1, dtype=torch.int32, device=x.device)
        cu[1:] = torch.cumsum(q_lens, dim=0)
        max_seq_len = int(q_lens.max().item())
        out = self.encoder_layer(
            src=x_pack,
            src_pos=pos_pack,
            cu_seq_len=cu,
            max_seq_len=max_seq_len,
        )
        x_proc = out.squeeze(0)
        x_rec = rearrange(x_proc, "(S C) D -> C S D", S=S)
        return x_rec

    def forward_cc_masked(
        self,
        x: torch.Tensor,               # (C, S, D)
        pos: torch.Tensor,             # (C, S, 2)
        mask: torch.Tensor,            # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Like forward_cc, but only processes unmasked tokens via varlen path.
        """
        x_seq = rearrange(x, "C S D -> S C D")
        pos_seq = rearrange(pos, "C S D -> S C D")
        mask_seq = rearrange(mask, "C S -> S C")

        mask_indices = get_non_zero_indices("MarkerAttention_cc_Masked_Mask_indices", ~mask_seq)
        x_false = x_seq[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_seq[mask_indices].unsqueeze(0) # (1, N, 2)

        # Build varlen based on channel-concat per sample
        seq_lens, max_seq_len = build_self_attention_bias_channel_concat(
            "MarkerAttention_cc_masked",
            mask_seq,
            tuple(channels_per_sample),
            use_true_as_query=False,
        )

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            cu_seq_len=seq_lens,
            max_seq_len=max_seq_len,
        )

        x_proc = out
        x_seq[mask_indices] = x_proc[0]
        x_rec = rearrange(x_seq, "S C D -> C S D")
        return x_rec


class FullAttentionEncoderBlock(nn.Module):
    """
    Encoder over the full (C × S) sequence per sample.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,         # (B, C, S, D)
        pos: torch.Tensor,       # (B, C, S, 2)
        mask: torch.Tensor,      # (B, C, S)
    ) -> torch.Tensor:
        """
        Masked full attention path (not implemented in varlen form).
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_cc(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full attention on concatenated (C × S) sequences per sample.

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list of channel counts per sample.

        Returns:
            x': (C, S, D) 
        """
        S = x.shape[1]
        # Per-sample lengths
        q_lens = torch.as_tensor([c * S for c in channels_per_sample], device=x.device, dtype=torch.int32)

        x_pack = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos_pack = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)

        # Cumulative lengths (prepend 0)
        cu = torch.zeros(q_lens.numel() + 1, dtype=torch.int32, device=x.device)
        cu[1:] = torch.cumsum(q_lens, dim=0)
        max_seq_len = int(q_lens.max().item())

        out = self.encoder_layer(
            src=x_pack,
            src_pos=pos_pack,
            cu_seq_len=cu,
            max_seq_len=max_seq_len,
        )

        x_proc = out.squeeze(0)
        x_rec = rearrange(x_proc, "(C S) D -> C S D", S=S)
        return x_rec

    def forward_cc_masked(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        mask: torch.Tensor,                    # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Masked full attention over concatenated (C × S) sequences per sample using varlen FlashAttention.
        Only the unmasked tokens are processed.
        """
        S = x.shape[1]

        x_flat = rearrange(x, "C S D -> (C S) D")
        pos_flat = rearrange(pos, "C S D -> (C S) D")
        mask_flat = rearrange(mask, "C S -> (C S)")

        # Build per-sample token counts (C_i * S)
        tokens_per_sample = tuple(int(c) * int(S) for c in channels_per_sample)

        mask_indices = get_non_zero_indices("FullAttention_cc_Masked_Mask_indices", ~mask_flat)
        x_false = x_flat[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_flat[mask_indices].unsqueeze(0) # (1, N, 2)

        # Varlen seq-lens from channel-concat mask
        seq_lens, max_seq_len = build_self_attention_bias_channel_concat(
            "FullAttention_cc_masked",
            mask_flat,
            tokens_per_sample,
            use_true_as_query=False,
        )

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            cu_seq_len=seq_lens,
            max_seq_len=max_seq_len,
        )

        x_proc = out
        x_flat[mask_indices] = x_proc[0]
        x_rec = rearrange(x_flat, "(C S) D -> C S D", S=S)
        return x_rec


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from x_query to x_keyval with positional embeddings.

    Args:
        model_dim (int): Model width.
        num_heads (int): Number of heads.
        dropout (float): Dropout prob.
        pos_type (str): Passed to MHAwithPosEmb (e.g., "learnable", "absolute", "rope").
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pos_type: Optional[str] = "learnable",
    ) -> None:
        super().__init__()
        self.attention_module = MHAwithPosEmb(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            inbuilt_pos_emb=pos_type,
        )

    def forward(
        self,
        x_query: torch.Tensor,             # (C_total, S, D)
        x_keyval: torch.Tensor,            # (C_total, S, D)
        pos: torch.Tensor,                 # (C_total, S, 2)
        multiplex_channels_per_sample: Sequence[int],
    ) -> torch.Tensor:
        """
        Pack sequences by sample (length = channels_i * S) and apply cross-attention.

        Returns:
            (C_total, S, D)
        """
        C_total, S, D = x_query.shape
        q_lens = torch.as_tensor([c * S for c in multiplex_channels_per_sample], device=x_query.device, dtype=torch.int32)

        # Pack as a single batch item
        _x_attn = rearrange(x_query, "C S D -> (C S) D").unsqueeze(0)  # (1, sumL, D)
        _prot = rearrange(x_keyval, "C S D -> (C S) D").unsqueeze(0)
        _pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)

        # Varlen cumulative sequence lengths
        cu = torch.zeros(q_lens.numel() + 1, dtype=torch.int32, device=x_query.device)
        cu[1:] = torch.cumsum(q_lens, dim=0)
        max_seq_len = int(q_lens.max().item())

        ca = self.attention_module(
            query=_x_attn,
            key=_prot,
            value=_prot,
            query_pos=_pos,
            key_pos=_pos,
            key_padding_mask=None,
            cu_seq_len=cu,
            max_seq_len=max_seq_len,
        )
        ca = rearrange(ca.squeeze(0), "(C S) D -> C S D", S=S)
        return ca


class PatchAttentionBlock(nn.Module):
    """
    Self-attention among patch summary tokens (first token per patch/channel).

    Useful after blocks that produce per-patch summary tokens you want to refine jointly.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, C, S, D)
            pos: (B, C, S, 2)

        Returns:
            x': (B, C, S, D) with updated first (summary) token per C.
        """
        B, C, S, D = x.shape

        # Take only the patch summary tokens: the first token across channels (index 0)
        ps = x[:, 0]      # (B, S, D)
        psp = pos[:, 0]   # (B, S, 2)

        # Pack into single long sequence for varlen attention
        ps = rearrange(ps, "B S D -> (B S) D")
        psp = rearrange(psp, "B S D -> (B S) D")

        # Build cu_seqlens = [0, S, 2S, ... B*S]
        # Use pure tensor ops (no Python list) for speed:
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * int(S)

        ps = self.encoder_layer(src=ps, src_pos=psp, max_seq_len=int(S), cu_seq_len=cu)
        ps = rearrange(ps, "(B S) D -> B S D", S=S)

        # Write back refined summary tokens
        x[:, 0] = ps
        return x

    def forward_masked(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Patch summary tokens are assumed to be always present; just call forward().
        """
        return self.forward(x, pos, **kwargs)

    def forward_cc(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        channels_per_sample: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Gather the first token of each channel group (per sample) and refine via attention.

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list with counts per sample; positions of first tokens
                                 are cumulative sums of these counts.
        """
        C, S, D = x.shape

        # Indices of the first token per channel group: cumulative sums across channel counts
        ps_position = np.cumsum(channels_per_sample)
        ps_position -= ps_position[0]  # shift to start at 0

        ps = x[ps_position]       # (B, S, D) where B = len(ps_position)
        psp = pos[ps_position]    # (B, S, 2)

        ps = rearrange(ps, "B S D -> (B S) D")
        psp = rearrange(psp, "B S D -> (B S) D")

        B_eff = len(ps_position)
        cu = torch.arange(B_eff + 1, device=x.device, dtype=torch.int32) * int(S)

        ps = self.encoder_layer(src=ps.unsqueeze(0), src_pos=psp.unsqueeze(0), max_seq_len=int(S), cu_seq_len=cu)
        ps = rearrange(ps.squeeze(0), "(B S) D -> B S D", S=S)

        x[ps_position] = ps
        return x

    def forward_cc_masked(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        channels_per_sample: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Same as forward_cc (summary tokens assumed always present).
        """
        return self.forward_cc(x, pos, channels_per_sample, **kwargs)
