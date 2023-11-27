from typing import Any, List


import numpy as np
import torch
from torch import nn
from torch import optim


class DDPM(nn.Module):
    def __init__(self, noise_scales: List[float], noise_estimator: nn.Module):
        super(DDPM, self).__init__()
        self.noise_scales = noise_scales
        self.noise_estimator = noise_estimator
        self.total_steps = len(noise_scales)
        self.accumulated_scales = [1 - self.noise_scales[0]]
        for i in range(1, self.total_steps):
            self.accumulated_scales.append(self.accumulated_scales[i - 1] * (1 - self.noise_scales[i]))

        self.noise_scales = nn.Parameter(torch.tensor(self.noise_scales).float(), requires_grad=False)
        self.accumulated_scales = nn.Parameter(torch.tensor(self.accumulated_scales).float(), requires_grad=False)

        # self.optimizer = optim.SGD(noise_estimator.parameters(), 0.1, momentum=0.9)
        self.optimizer = optim.Adam(noise_estimator.parameters())

    def train_one_batch(self, xs: torch.Tensor, steps: torch.Tensor):
        noise = torch.normal(0, 1, xs.shape, device=xs.device)

        # The noisy x at step t
        xs_t = torch.sqrt(self.accumulated_scales[steps]).view(-1, 1) * xs + \
               (torch.sqrt(1 - self.accumulated_scales[steps])).view(-1, 1) * noise
        predicted_noise = self.noise_estimator(xs_t, steps)
        loss = torch.mean(torch.square(predicted_noise - noise))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sample(self, ys: torch.Tensor):
        with torch.no_grad():
            for i in range(len(self.noise_scales) - 1, -1, -1):
                step_tensor = torch.ones((ys.shape[0],), device=ys.device, dtype=torch.long) * i

                if i != 0:
                    posterion_std = torch.sqrt(self.noise_scales[i] * (1 - self.accumulated_scales[i - 1]) / (1 - self.accumulated_scales[i]))
                else:
                    posterion_std = 0

                ys = (1 / torch.sqrt(1 - self.noise_scales[i])) * \
                     (ys - self.noise_scales[i]/torch.sqrt(1 - self.accumulated_scales[i]) * self.noise_estimator(ys, step_tensor))\
                     + posterion_std * torch.normal(0, 1, ys.shape, device=ys.device)
        return ys
