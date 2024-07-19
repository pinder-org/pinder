from __future__ import annotations
import math
from pathlib import Path

from PIL import Image as pimage
from PIL.Image import Image

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from numpy.typing import NDArray


rcParams["font.weight"] = "bold"
# rcParams['font.sans-serif'] = ['calibri']


def resize_img_to_array(
    img: Image,
    img_shape: tuple[int, int] = (244, 244),
) -> NDArray[np.int_]:
    img_array = np.array(
        img.resize(
            img_shape,
            pimage.LANCZOS,
        )
    )
    return img_array


def image_grid(
    fn_images: list[Path],
    text: list[str] = [],
    pdb_id: str | None = None,
    output_png: Path | None = None,
    per_row: int = 4,
    img_width: int = 2432,
    img_height: int = 1462,
    left: float | None = None,
    right: float | None = None,
    top: float | None = 0.88,
    bottom: float | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    figsize: tuple[int, int] = (22, 10),
) -> None:
    """fn_images is a list of image paths.
    text is a list of annotations.
    top is how many images you want to display
    per_row is the number of images to show per row.
    """
    n_rows = math.ceil(len(fn_images) / per_row)
    fig, ax = plt.subplots(n_rows, per_row, sharex="col", sharey="row", figsize=figsize)
    row_idx = 0
    plot_idx = 0
    for row_idx in range(len(ax)):
        for col_idx in range(per_row):
            if per_row == 1:
                ax_slice = ax[row_idx]
            elif per_row == len(ax):
                ax_slice = ax[col_idx]
            else:
                ax_slice = ax[row_idx][col_idx]
            if plot_idx >= len(fn_images):
                ax_slice.axis("off")
            else:
                image = pimage.open(fn_images[plot_idx])
                image_arr = resize_img_to_array(
                    image, img_shape=(img_width, img_height)
                )
                ax_slice.imshow(image_arr)
                ax_slice.axis("off")
                if text:
                    ax_slice.annotate(
                        text[plot_idx],
                        (0, 0),
                        (0, -32),
                        xycoords="axes fraction",
                        textcoords="offset points",
                        fontsize=15,
                        va="top",
                    )
            plot_idx += 1

    if pdb_id:
        fig.suptitle(f"PDB ID: {pdb_id}", size=18, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace
    )
    if output_png:
        plt.savefig(str(output_png), dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
