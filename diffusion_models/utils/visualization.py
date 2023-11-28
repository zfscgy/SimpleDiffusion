import matplotlib.pyplot as plt
import numpy as np


def show_image(img: np.ndarray, n_batch_dims: int = 0, save_to: str = None):
    colormap = None
    plt.axis('off')
    if len(img.shape) - n_batch_dims == 2:
        colormap = 'gray'
    if n_batch_dims == 0:
        plt.imshow(img, cmap=colormap)
        if save_to:
            plt.savefig(save_to)
        plt.show()
    else:
        if n_batch_dims == 1:
            img = img[np.newaxis, ...]

        n_rows, n_cols = img.shape[:2]

        for i in range(n_rows):
            for j in range(n_cols):
                plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
                plt.imshow(img[i, j], cmap=colormap)
                plt.axis('off')

        if save_to:
            plt.savefig(save_to)
        plt.show()
