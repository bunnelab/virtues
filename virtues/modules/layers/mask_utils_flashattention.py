from modules.layers.cache import LRUCache
import torch
from typing import Optional, Tuple, List, Union

SELF_ATTENTION_BIAS_CACHE = LRUCache(cache_len=20)



@torch.no_grad()
def calculate_seq_len_cumsums_and_max(split_mask: Union[Tuple[torch.Tensor], torch.Tensor], use_true_as_query: bool = True,
                           filter_zero_len_seq: bool = True, prepend_zero: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Calculate cumulative sequence lengths from a binary mask.
    Args:
        split_mask (Union[Tuple[torch.Tensor], torch.Tensor]): Boolean tensor of shape (..., S) where True indicates valid tokens.
        use_true_as_query (bool): If True, use True values as queries; otherwise, use False values as queries.
        filter_zero_len_seq (bool): If True, filter out sequences with zero length.
        prepend_zero (bool): If True, prepend a zero to the cumulative lengths.
    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - seq_lens (torch.Tensor): 1D tensor of cumulative sequence lengths for Flash
            - max_seq_len (int): Maximum sequence length in the batch.
    """
    calc_sum = lambda x: x.sum(-1) if use_true_as_query else (~x).sum(-1)
    if isinstance(split_mask, tuple):
        seq_lens = torch.stack(
            [calc_sum(m) for m in split_mask], 
            dim=-1
        )
    else:
        seq_lens = calc_sum(split_mask)
    if filter_zero_len_seq:
        seq_lens = seq_lens[seq_lens != 0]
    if prepend_zero:
        seq_lens = torch.cat([
            torch.zeros(1, dtype=seq_lens.dtype, device=seq_lens.device),
            seq_lens
        ])
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    return seq_lens.cumsum(dim=0, dtype=torch.int32), max_seq_len


@torch.no_grad()
def build_self_attention_bias(
        cache_key: str,
        split_mask: torch.Tensor,
        use_true_as_query: bool = True,
        device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, int]:
    """
    Build self-attention bias for FlashAttention from a binary mask. Checks in cache first.
    Args:
        cache_key (str): Key to use for caching the computed bias.
        split_mask (torch.Tensor): Boolean tensor of shape (..., S) where True indicates valid tokens.
        use_true_as_query (bool): If True, use True values as queries; otherwise, use False values as queries.
        device (Optional[torch.device]): Device to place the output tensors on. Defaults to the device of `split_mask`.
    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - seq_lens (torch.Tensor): 1D tensor of cumulative sequence lengths for FlashAttention.
            - max_seq_len (int): Maximum sequence length in the batch.
    """
    if device is None:
        device = split_mask.device

    cache_result = SELF_ATTENTION_BIAS_CACHE[cache_key]
    if cache_result is not None:
        cached_seq_lens, cached_max_seq_len = cache_result
        return (cached_seq_lens, cached_max_seq_len)

    
    seq_lens, max_seq_len = calculate_seq_len_cumsums_and_max(
        split_mask, use_true_as_query=use_true_as_query, filter_zero_len_seq=True, prepend_zero=True
    )

    # Cache result
    SELF_ATTENTION_BIAS_CACHE[cache_key] = (seq_lens, max_seq_len)
    return (seq_lens, max_seq_len)







@torch.no_grad()
def build_self_attention_bias_channel_concat(cache_key: str, split_mask: torch.Tensor,
                                             tokens_per_sequence: List[int],
                                             use_true_as_query: bool = True,
                                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, int]:
    
    """
    Build self-attention bias for FlashAttention when channels of different samples
    are concatenated along the final dimension and should not attend to each other.
    Checks in cache first.

    Args:
        cache_key (str): Key to use for caching the computed bias.
        split_mask (torch.Tensor): Boolean tensor of shape (..., S) where True indicates valid tokens.
        tokens_per_sequence (List[int]): Number of channels per sample; must sum up to S across samples.
        use_true_as_query (bool): If True, use True values as queries; otherwise, use False values as queries.
        device (Optional[torch.device]): Device to place the output tensors on. Defaults to the device of `split_mask`.

    Returns:
        Tuple[torch.Tensor, int]:
            - seq_lens (torch.Tensor): 1D tensor of cumulative sequence lengths for FlashAttention.
            - max_seq_len (int): Maximum sequence length across all samples.
    """
    if device is None:
        device = split_mask.device

    cache_result = SELF_ATTENTION_BIAS_CACHE[cache_key]
    if cache_result is not None:
        cached_seq_lens, cached_max_seq_len = cache_result
        return (cached_seq_lens, cached_max_seq_len)
    
    # assert split_mask.shape[-1] != tokens_per_sequence.sum().item(), \
    #     f"tokens_per_sequence ({tokens_per_sequence}) must sum up to the final dimension of split_mask ({split_mask.shape[-1]})."

    split_masks = split_mask.split(tokens_per_sequence, dim=-1)
    seq_lens, max_seq_len = calculate_seq_len_cumsums_and_max(
        split_masks, use_true_as_query=use_true_as_query, filter_zero_len_seq=True, prepend_zero=True
    )
    # Cache result
    SELF_ATTENTION_BIAS_CACHE[cache_key] = (seq_lens, max_seq_len)
    return (seq_lens, max_seq_len)


@torch.no_grad()
def get_non_zero_indices(cache_key: str, x:torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Get indices of non-zero elements in a tensor, with caching.
    Args:
        cache_key (str): Key to use for caching the computed indices.
        x (torch.Tensor): Input tensor.
    Returns:
        tuple[torch.Tensor]: Indices of non-zero elements.
    """
    cache_result = SELF_ATTENTION_BIAS_CACHE[cache_key]
    if cache_result is not None:
        return cache_result

    non_zero_indices = torch.nonzero(x, as_tuple=True)
    SELF_ATTENTION_BIAS_CACHE[cache_key] = non_zero_indices
    return non_zero_indices