import torch
import numpy as np

CELL_PALETTE = {
    'Tumor': "#1A254B",
    'Fibroblasts /Stroma': "#A7BED3",
    'Myeloid': "#6C3483",
    'CD4 T cell': "#F08b8f",
    'CD8 T cell': "#ed3941",
    'B cell': "#a15764",
    'Vessel / Endo.': "#2B50AA",
    "None": "#FFFFFF",
    'Unknown': "#d0d0d0"
}

def transform_mask_to_RGB(mask, id_to_rgba):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for id, color in id_to_rgba.items():
        rgb[mask == id] = color
    return rgb

def parse_color(value):
    """
    Supports:
      - Hex strings: "#RRGGBB", "RRGGBB", "#RGB", "RGB"
      - Integer RGB: (255, 128, 0) or [255, 128, 0]
      - Float RGB in [0,1]: (1.0, 0.5, 0.0) or [1.0, 0.5, 0.0]

    Returns:
      (r, g, b) as integers in [0,255]
    """

    # Hex string input
    if isinstance(value, str):
        value = value.lstrip("#")

        if len(value) == 3:
            value = "".join(c * 2 for c in value)

        if len(value) != 6:
            raise ValueError("Invalid hex color string")

        return np.array(tuple(int(value[i:i+2], 16) for i in (0, 2, 4)))

    # RGB list/tuple input
    if isinstance(value, (list, tuple)) and len(value) == 3:
        # Float RGB [0,1]
        if all(isinstance(x, float) for x in value):
            if not all(0.0 <= x <= 1.0 for x in value):
                raise ValueError("Float RGB values must be in [0,1]")
            return np.array(tuple(round(x * 255) for x in value))

        # Int RGB [0,255]
        if all(isinstance(x, int) for x in value):
            if not all(0 <= x <= 255 for x in value):
                raise ValueError("Integer RGB values must be in [0,255]")
            return np.array(tuple(value))

    raise TypeError("Unsupported color format")

def plot_mask(mask, id_to_name, name_to_color, ax=None):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    id_to_rgb = {id: parse_color(name_to_color[name]) for id, name in id_to_name.items()}
    mask_rgb = transform_mask_to_RGB(mask, id_to_rgb)

    values = sorted(id_to_name.keys())
    labels = [id_to_name[v] for v in values]
    colors = [name_to_color[id_to_name[v]] for v in values]
    # cmap = ListedColormap(colors)
    # if ax is None:
    #     plt.imshow(mask, cmap=cmap, vmin=values[0], vmax=values[-1])
    # else:
    #     ax.imshow(mask, cmap=cmap, vmin=values[0], vmax=values[-1])
    if ax is None:
        plt.imshow(mask_rgb)
    else:
        ax.imshow(mask_rgb)
