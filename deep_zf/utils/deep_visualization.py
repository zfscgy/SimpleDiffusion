import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from deep_zf.convert import Convert



def tsne(feature_extractor: nn.Module, data_loader: DataLoader, n_samples: int = 3000, save_png: str = None):
    all_hs = []
    all_ys = []
    i = 0
    for xs, ys in data_loader:
        with torch.no_grad():
            all_hs.append(feature_extractor(Convert.to_tensor(xs)).cpu().numpy())
            all_ys.append(ys.numpy())
        i += 1
        if i > n_samples / data_loader.batch_size:
            break

    all_hs = np.concatenate(all_hs, axis=0)
    all_ys = np.concatenate(all_ys, axis=0)

    # print("hs shape", all_hs.shape)
    # print("ys shape", all_ys.shape)

    tsne = TSNE(2, random_state=19260817)
    all_rs = tsne.fit_transform(all_hs)

    for c in range(np.max(all_ys) + 1):
        rs = all_rs[all_ys == c]
        plt.scatter(rs[:, 0], rs[:, 1], marker="x", label=str(c))
    plt.axis('off')
    if save_png:
        plt.savefig(save_png, bbox_inches='tight')
    plt.show()
    plt.clf()
