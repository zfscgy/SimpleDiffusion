from typing import Callable
from torch import nn


class LambdaLayer(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func: Callable):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)