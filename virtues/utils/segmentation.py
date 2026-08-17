import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def remove_small_cells(prediction: torch.Tensor | np.ndarray, min_cell_size: int = 15) -> np.ndarray:
    labels, counts = np.unique(prediction, return_counts=True)
    small_cells = labels[counts < min_cell_size]
    prediction = np.where(np.isin(prediction, small_cells), 0, prediction)
    labels, inverse = np.unique(prediction, return_inverse=True)
    relabelled = inverse.reshape(prediction.shape)
    return relabelled.astype(np.int32)

def assign_cell_types(instance_prediction: np.ndarray, semantic_prediction: np.ndarray) -> np.ndarray:
    inst = instance_prediction.ravel()
    sem = semantic_prediction.ravel().astype(np.intp)

    ids, inv = np.unique(inst, return_inverse=True)
    inv = inv.ravel()
    n_types = int(sem.max()) + 1 if sem.size else 1

    # counts[i, t] = number of pixels of instance ids[i] with semantic label t
    counts = np.bincount(inv * n_types + sem, minlength=ids.size * n_types)
    counts = counts.reshape(ids.size, n_types)

    types_per_id = counts.argmax(axis=1).astype(np.int32)
    types_per_id[ids == 0] = 0

    return types_per_id[inv].reshape(instance_prediction.shape)