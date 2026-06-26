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