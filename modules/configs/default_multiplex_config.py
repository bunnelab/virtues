import torch
from typing import Dict, Any

DEFAULT_MULTIPLEX_CONFIG: Dict[str, Any] = {
        'prior_bias_embeddings': torch.zeros(512, 768),  # Example default tensor
        'prior_bias_embedding_type': 'zero',
        'patch_size': 8,
        'model_dim': 512,
        'feedforward_dim': 1024,
        'encoder_pattern': 'hvhvhvhv',
        'num_encoder_heads': 8,
        'decoder_pattern': 'hvhvhvhv', 
        'num_decoder_heads': 8,
        'num_hidden_layers': 12,
        'positional_embedding_type': 'sinusoidal',
        'dropout': 0.1,
        'group_layers': 1.0,
        'norm_after_encoder_decoder': False,
        'prior_bias_embedding_fusion_type': 'add',
    }