import os
import torch
import numpy as np
import random

def is_rank0():
    """
    Checks if the current process is rank 0 in a distributed setting.
    """
    return os.environ.get('RANK', '0') == '0'

def set_seed(seed : int):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_marker_embedding_dict(embedding_dir : str):
    """
    Returns a dictionary mapping protein IDs to their index in the marker embedding matrix of `load_marker_embeddings`.
    embedding_dir: directory containing marker embeddings in .pt format
    """
    embedding_dict = {}
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Could not find embedding_dir {embedding_dir}")
    files = os.listdir(embedding_dir)
    files = list(sorted(files))
    index = 0
    for file in files:
        if file.endswith(".pt"):
            protein_id = file.removesuffix(".pt")
            embedding_dict[protein_id] = index
            index += 1
    return embedding_dict

def load_marker_embeddings(embedding_dir : str):
    """
    Loads all marker embeddings from the specified directory.
    embedding_dir: directory containing marker embeddings in .pt format
    Returns a tensor of shape (num_markers, embedding_dim)
    """
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Could not find embedding_dir {embedding_dir}")
    files = os.listdir(embedding_dir)
    files = sorted(files)
    embeddings = []
    for file in files:
        if file.endswith(".pt"):
            embeddings.append(torch.load(os.path.join(embedding_dir, file), weights_only=True))
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings

def to_device(x, device):
    """
    Moves tensor(s) to the specified device.
    x: tensor or list/tuple of tensors
    device: target device (e.g., 'cuda' or 'cpu')
    """
    if isinstance(x, list):
        return [to_device(i, device) for i in x]
    if isinstance(x, tuple):
        return [to_device(i, device) for i in x]
    if x is None:
        return None
    else:
        return x.to(device)
    

