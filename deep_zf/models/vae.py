from typing import Any, Callable
import io
import torch
from torch import nn
from torch.optim import Optimizer, Adam

from deep_zf.workflow.train import ZModel


class VAE(ZModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, get_optimizer: Callable[[Any], Optimizer], loss_coef: float = 1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        get_optimizer = get_optimizer or Adam
        self.optimizer = get_optimizer(list(self.decoder.parameters()) + list(self.decoder.parameters()))
        self.loss_coef = loss_coef

    def forward(self, xs: torch.Tensor) -> Any:
        h_para = self.encoder(xs)
        hidden_dim = h_para.shape[1]
        h_means = h_para[:, :, 0]
        h_stds = h_para[:, :, 1]
        normal = torch.normal(0, 1, [hidden_dim]).to(xs.device)
        h_sample = h_means + h_stds * normal
        return self.decoder(h_sample)

    def compute_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        h_para = self.encoder(xs)
        hidden_dim = h_para.shape[1]
        h_means = h_para[:, :, 0]
        h_stds = h_para[:, :, 1]
        normal = torch.normal(0, 1, [hidden_dim]).to(xs.device)
        h_sample = h_means + h_stds * normal
        pred_ys = self.decoder(h_sample)

        loss_rec = torch.sum(torch.square(pred_ys - ys))  # Reconstruction loss
        loss_reg = torch.sum(h_means ** 2 + h_stds ** 2 - torch.log(h_stds))  # Regularization loss
        self.batch_loss_record.append([loss_rec.item(), loss_reg.item()])
        return loss_rec + self.loss_coef * loss_reg

    def train_step(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_loss(xs, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        binary_stream = io.BytesIO()
        torch.save(nn.Sequential(self.encoder, self.decoder).state_dict(), binary_stream)
        binary_stream.seek(0)
        return torch.load(binary_stream)

    def load(self, saved_obj):
        nn.Sequential(self.encoder, self.decoder).load_state_dict(saved_obj)

    def set_train(self, mode: bool):
        self.encoder.train(mode)
        self.decoder.train(mode)
