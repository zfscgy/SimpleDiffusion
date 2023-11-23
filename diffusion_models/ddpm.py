from typing import Any, List

import torch
from torch import nn
from torch import optim



class DDPM:
    def __init__(self, noise_scales: List[float], noise_estimator: nn.Module):
        super(DDPM, self).__init__()
        self.noise_scales = noise_scales
        self.noise_estimator = noise_estimator
        self.total_steps = len(noise_scales)
        self.accumulated_scales = [1 - self.noise_scales[0]]
        for i in range(1, self.total_steps):
            self.accumulated_scales.append(self.accumulated_scales[-1] * (1 - self.noise_scales[i]))

        self.optimizer = optim.SGD(noise_estimator.parameters(), 0.01, momentum=0.9)

    def train_one_batch(self, xs: torch.Tensor, step: int):
        if step > len(self.accumulated_scales):
            raise ValueError("Invalid step.")

        noise = torch.normal(0, 1, xs.shape)

        # The noisy x at step t
        xs_t = torch.sqrt(self.accumulated_scales[step]) * xs + (1 - torch.sqrt(self.accumulated_scales[step])) * noise
        predicted_noise = self.noise_estimator(xs_t, step)
        loss = torch.sum(torch.square(predicted_noise - noise)) / xs.shape[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, ys: torch.Tensor):
        for i in range(len(self.noise_scales) - 1, -1, -1):
            ys = (1 / torch.sqrt(self.accumulated_scales[i])) * \
                 (ys - self.noise_scales[i]/torch.sqrt(1 - self.accumulated_scales[i]) * self.noise_estimator(ys, i))\
                 + torch.sqrt(self.noise_scales[i]) * torch.normal(0, 1, ys.shape)

        return ys
