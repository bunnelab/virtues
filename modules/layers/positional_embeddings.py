import torch
import torch.nn as nn
from torch import LongTensor
from typing import Optional, Tuple, List, Union


class PositionalEmbedding2D(nn.Module):
    """
    Sinusoidal 2D positional embedding module.

    This implements a 2D extension of the standard Transformer-style sinusoidal
    embeddings. Each position in (row, col) space gets mapped into a fixed
    embedding of dimension `model_dim`, which is added to the input features.

    Args:
        model_dim (int): Dimension of the modules. Must be divisible by 4
            (since half is used for rows and half for columns, each with
            sin/cos pairs).
        max_width_or_height (int): Maximum size for width or height dimension.
            Determines the number of distinct positions that can be embedded.
            Defaults to 1200.
        temperature (float): Temperature scaling factor used in frequency
            calculation (as in the original Transformer positional encoding).
            Defaults to 10000.0.
    """
    def __init__(
            self,
            model_dim: int,
            max_width_or_height: int = 1200,
            temperature: float = 10000.0
    ) -> None:
        super().__init__()
        if model_dim % 4 != 0:
            raise ValueError("model_dim must be divisible by 4 for 2D positional embeddings.")
        self.model_dim = model_dim
        self.max_width_or_height = max_width_or_height
        self.temperature = temperature
        
        # Precompute the positional encodings for efficiency
        self.register_buffer('positional_encoding', self._create_positional_encoding(), persistent=False)

    def _create_positional_encoding(self) -> torch.Tensor:
        dim_pe = self.model_dim // 2
        positions = torch.arange(self.max_width_or_height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_pe, 2).float() * (-torch.log(torch.tensor(self.temperature)) / dim_pe))
        pe = torch.zeros(self.max_width_or_height, dim_pe)
        pos = positions * div_term
        pe[:, 0::2] = torch.sin(pos)
        pe[:, 1::2] = torch.cos(pos)
        return pe # Shape: (max_width_or_height, model_dim/2)        
    

    def forward(self, x: torch.Tensor, positions: Optional[LongTensor] = None) -> torch.Tensor:
        """
        Forward pass to add positional embeddings to input features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where
                B is batch size, C is number of channels (should equal model_dim),
                H is height, and W is width.
            positions (Optional[LongTensor]): Optional tensor of shape (B, 2)
                specifying the (row, col) positions for each item in the batch.
                If None, assumes positions are centered in the feature map.

        Returns:
            torch.Tensor: Output tensor of same shape as input with positional
                embeddings added.
        """
        if x.dim() != 4 or x.size(1) != self.model_dim:
            raise ValueError(f"Input tensor must have shape (B, {self.model_dim}, H, W).")
        
        B, C, H, W = x.shape
        
        if positions is None:
            # Default to center positions
            positions = torch.tensor([[H // 2, W // 2]] * B, device=x.device)
        elif positions.shape != (B, 2):
            raise ValueError("positions tensor must have shape (B, 2).")
        
        row_pos_emb = self.positional_encoding[positions.select(dim=-1, index=0)]
        col_pos_emb = self.positional_encoding[positions.select(dim=-1, index=1)]
        pos_emb = torch.cat([row_pos_emb, col_pos_emb], dim=-1)
        return x + pos_emb
    

class RotaryPositionalEmbedding1D(nn.Module):
    """
    Implements rotary positional embeddings in 1D
    Args:
        model_dim (int): Dimension of the modules. Must be even.
        max_seq_len (int): Maximum sequence length to support. Defaults to 1200.
        temperature (float): Temperature scaling factor used in frequency
    """

    def __init__(self, model_dim: int, max_seq_len: int = 1200, temperature: float = 10000.0) -> None:
        super().__init__()
        if model_dim % 2 != 0:
            raise ValueError("model_dim must be even for rotary positional embeddings.")
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        
        self._create_sin_cos_buffers()

    def _create_sin_cos_buffers(self) -> None:
        possible_positions = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-torch.log(torch.tensor(self.temperature)) / self.model_dim))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        sin = torch.concat([sin, sin], dim=-1)
        self.register_buffer('sin', sin)
        cos = torch.cos(pos)
        cos = torch.concat([cos, cos], dim=-1)
        self.register_buffer('cos', cos)

    def _invert_negate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to rotate half the dimensions by negating the second half.
        """
        return torch.cat(
            [-x[..., self.model_dim // 2:], x[..., :self.model_dim // 2]], 
            dim=-1
        )
    
    def forward(self, x: torch.Tensor, positions: Optional[LongTensor] = None) -> torch.Tensor:
        """
        Forward pass to apply rotary positional embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (..., model_dim)
            positions (Optional[LongTensor]): Optional 1D tensor of shape (S,)
                specifying the positions to use for each token in the sequence.
                If None, assumes positions are [0, 1, ..., S-1] where S is the
                sequence length (x.size(-2)).
        """
        if x.dim() < 2 or x.size(-1) != self.model_dim:
            raise ValueError(f"Input tensor must have last dimension of size {self.model_dim}.")
        
        if positions is None:
            seq_len = x.size(-2)
            if seq_len > self.max_seq_len:
                raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")
            positions = torch.arange(seq_len, device=x.device)
        
        x = x * self.cos[positions] + self._invert_negate(x) * self.sin[positions]
        return x
    

class RotaryPositionalEmbedding2D(nn.Module):
    """
    Implements rotary positional embeddings in 2D.
    Args:
        model_dim (int): Dimension of the modules. Must be divisible by 4.
        max_width_or_height (int): Maximum size for width or height dimension.
            Determines the number of distinct positions that can be embedded.
            Defaults to 1200.
        temperature (float): Temperature scaling factor used in frequency
            calculation (as in the original Transformer positional encoding).
            Defaults to 10000.0.
    """

    def __init__(self, model_dim: int, max_width_or_height: int = 1200, temperature: float = 10000.0) -> None:
        super().__init__()
        if model_dim % 4 != 0:
            raise ValueError("model_dim must be divisible by 4 for 2D rotary positional embeddings.")
        self.model_dim = model_dim
        self.max_width_or_height = max_width_or_height
        self.temperature = temperature
        self.rope1d = RotaryPositionalEmbedding1D(model_dim // 2, max_width_or_height, temperature)

    def forward(self, x: torch.Tensor, positions: LongTensor) -> torch.Tensor:
        """
        Forward pass to apply 2D rotary positional embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where
                B is batch size, C is number of channels (should equal model_dim),
                H is height, and W is width.
            positions (LongTensor): Tensor of shape (B, 2) specifying the (row, col)
                positions for each item in the batch.
        """
        
        d = self.model_dim // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = self.rope1d(x1, positions.select(dim=-1, index=0))
        x2 = self.rope1d(x2, positions.select(dim=-1, index=1))
        return torch.cat([x1, x2], dim=-1)
    

class LearnablePositionalEmbedding2D(nn.Module):
    """
    Learnable 2D positional embedding module.

    Unlike sinusoidal embeddings, this module learns a parameterized embedding
    for each (row, col) position up to a maximum grid size.

    Args:
        model_dim (int): Dimension of the modules. Each embedding vector will have this size.
        max_pos (int): Maximum number of positions along each axis (height/width).
            The embedding table will therefore be of shape (max_pos, max_pos, model_dim).
            Defaults to 100.
    """

    def __init__(
        self,
        model_dim: int,
        max_pos: int = 100
    ) -> None:
        super().__init__()
        # Initialize learnable parameters with small random values
        self.pos_embeddings = nn.Parameter(
            torch.randn(max_pos, max_pos, model_dim) / (model_dim ** 2)
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: LongTensor
    ) -> torch.Tensor:
        """
        Add learnable 2D positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., model_dim).
            pos (LongTensor): Tensor of shape (..., 2) containing row and
                column indices for each element in `x`.

        Returns:
            torch.Tensor: Tensor of shape (..., model_dim) with positional
            embeddings added.
        """
        # Gather positional embeddings based on row/col indices
        row_idx = pos[..., 0]
        col_idx = pos[..., 1]
        to_add = self.pos_embeddings[row_idx, col_idx]

        return x + to_add