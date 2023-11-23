from typing import List, Callable, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SubDataset(Dataset):
    def __init__(self, original_dataset: Dataset, indices: List[int], repitition: int = 1):
        indices = list(indices)
        self.original_dataset = original_dataset
        self.randomize_factor = repitition
        self.indices = indices * repitition
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset: Dataset, portion: float=0.9):
    dataset_size = len(dataset)
    random_indices = np.random.permutation(dataset_size)
    return \
        SubDataset(dataset, random_indices[:round(dataset_size * portion)]), \
        SubDataset(dataset, random_indices[round(dataset_size * portion):])


class TransformationDataset(Dataset):
    def __init__(self, original_dataset: Dataset, transformation: Callable):
        self.original_dataset = original_dataset
        self.transformation = transformation

    def __getitem__(self, index: int):
        data_batch = self.original_dataset[index]
        return self.transformation(data_batch)

    def __len__(self):
        return len(self.original_dataset)


class IdentityDataset(TransformationDataset):
    """
    Making Y = X, ignore the actual label. Can be used for autoencoder.
    """
    def __init__(self, original_dataset: Dataset):
        super(IdentityDataset, self).__init__(original_dataset, lambda d: (d[0], d[0]))

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        return x, x

    def __len__(self):
        return len(self.original_dataset)


class LabelDPDataset(Dataset):
    def __init__(self, original_dataset: Dataset, label_flip_prob: float):
        self.original_dataset = original_dataset
        dataset_size = len(original_dataset)
        randomized_indices = np.random.choice(dataset_size, int(dataset_size * label_flip_prob))

        ys = []
        for _, y in self.original_dataset:
            ys.append(y)
        ys = np.array(ys)

        largest_ys = np.max(ys)
        ys[randomized_indices] = np.random.randint(0, largest_ys + 1, len(randomized_indices))
        self.label = ys

    def __getitem__(self, index):
        x, _ = self.original_dataset[index]
        return x, self.label[index]

    def __len__(self):
        return len(self.original_dataset)


def filter_dataset(original_dataset: Dataset, filter: Callable[[Any, Any], bool]):
    indices = []
    for i, (x, y) in enumerate(original_dataset):
        if filter(x, y):
            indices.append(i)
    return SubDataset(original_dataset, indices)


def get_equal_class_subset(dataset: Dataset, samples_per_class, repetition: int = 1):
    class_indices = dict()
    for i, (x, y) in enumerate(dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if y not in class_indices:
            class_indices[y] = []
        class_indices[y].append(i)

    indices = []
    for c in class_indices:
        indices.extend(np.random.choice(class_indices[c], samples_per_class))

    return SubDataset(dataset, indices, repetition)


def get_subclass_dataset(dataset: Dataset, classes: List[int]):
    class_indices = dict()
    for i, (x, y) in enumerate(dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if y not in class_indices:
            class_indices[y] = []
        class_indices[y].append(i)

    indices = []
    for c in classes:
        indices.extend(class_indices[c])

    def label_transform(d):
        _, y = d
        return _, classes.index(y)

    return TransformationDataset(SubDataset(dataset, indices), label_transform)


def get_label_extension_dataset(dataset: Dataset, new_label: List[int]):
    ys = []
    for x, y in dataset:
        ys.append(y)

    for i, y in enumerate(ys):
        if isinstance(y, List):
            y.append(new_label[i])
        else:
            ys[i] = [y, new_label[i]]

    class LabelChangeDataset:
        def __init__(self, original_dataset: Dataset, new_label: List):
            self.original_dataset = original_dataset
            self.new_label = new_label

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, index: int):
            return self.original_dataset[index][0], new_label[index]

    return LabelChangeDataset(dataset, ys)
