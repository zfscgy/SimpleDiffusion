import matplotlib.pyplot as plt
import numpy as np


def show_image(img: np.ndarray, save_to: str = None):
    colormap = None
    plt.axis('off')
    if len(img.shape) == 2:
        colormap = 'gray'
    if save_to:
        plt.imsave(save_to, img, cmap=colormap)
    plt.imshow(img, cmap=colormap)
    plt.show()
