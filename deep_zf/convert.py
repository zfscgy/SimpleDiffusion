from typing import Union, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from deep_zf.config import GlobalConfig
from deep_zf.data.utils import TransformationDataset


class Convert:
    @staticmethod
    def to_numpy(x: Union[List, torch.Tensor]) -> Union[np.ndarray, List]:
        if isinstance(x, torch.Tensor):
            try:
                return x.cpu().detach().numpy()
            except:
                pass
            return x.cpu().numpy()
        else:  # List
            return [Convert.to_numpy(xx) for xx in x]

    @staticmethod
    def to_tensor(x: Union[List, torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, List]:
        if isinstance(x, List):
            return [Convert.to_tensor(e) for e in x]
        if isinstance(x, torch.Tensor):
            return x.to(GlobalConfig.device)
        else:
            return torch.tensor(x).to(GlobalConfig.device)

    @staticmethod
    def model_to_device(model: Union[nn.Module, List[nn.Module]]) -> Union[nn.Module, List]:
        if isinstance(model, nn.Module):
            return model.to(GlobalConfig.device)
        else:
            return [Convert.model_to_device(m) for m in model]

    @staticmethod
    def model_to_cpu(model: Union[nn.Module, List[nn.Module]]) -> Union[nn.Module, List]:
        if isinstance(model, nn.Module):
            return model.to('cpu')
        else:
            return [Convert.model_to_cpu(m) for m in model]

