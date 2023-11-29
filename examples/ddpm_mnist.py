from typing import Callable, List, Tuple
from functools import partial

import os
from pathlib import Path



from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
DataLoader = partial(DataLoader, num_workers=4, shuffle=True)

from deep_zf.convert import Convert
from deep_zf.data.datasets import Mnist
from deep_zf.models.utils import LambdaLayer


from diffusion_models.ddpm import DDPM
from diffusion_models.utils.visualization import show_image

noise_scales = np.arange(0.001, 0.0301, 0.0001)
print(f"Steps: {len(noise_scales)}")
print(f"Final original image scale: {np.prod(1 - noise_scales):.5f}")


class SimpleResidualLayer(nn.Module):
    def __init__(self, BasicBlock: Callable[[], nn.Module]):
        super(SimpleResidualLayer, self).__init__()
        self.basic_block1 = BasicBlock()
        self.basic_block2 = BasicBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.basic_block2(self.basic_block1(x))


class DownSampleLayer(nn.Module):
    def __init__(self, in_channel: int, output_shape: List[int]):
        """

        :param in_channel:
        :param output_shape: [n_channels, height, width]
        """
        out_channel = output_shape[0]
        super(DownSampleLayer, self).__init__()
        self.residual1 = SimpleResidualLayer(lambda: nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()))
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU())
        self.layer_norm1 = nn.LayerNorm(output_shape)
        self.residual2 = SimpleResidualLayer(lambda: nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()))
        self.layer_norm2 = nn.LayerNorm(output_shape)

    def forward(self, xs: torch.Tensor):
        h0 = self.residual1(xs)
        h1 = self.layer_norm2(self.down_conv(h0))
        h2 = self.layer_norm2(self.residual2(h1))
        return h2, h0


class UpSampleLayer(nn.Module):
    def __init__(self, in_channel: int, output_shape: List[int]):
        """

        :param in_channel:
        :param output_shape: [n_channels, height, width]
        """
        out_channel = output_shape[0]
        super(UpSampleLayer, self).__init__()
        self.residual1 = SimpleResidualLayer(lambda: nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()))
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2))
        self.layer_norm1 = nn.LayerNorm(output_shape)
        self.residual2 = SimpleResidualLayer(lambda: nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()))
        self.layer_norm2 = nn.LayerNorm(output_shape)

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.residual1(xs)
        h1 = self.layer_norm2(self.up_conv(h0)[..., 1:, 1:])
        h2 = self.layer_norm2(self.residual2(h1))
        return h2



class UNET_MnistDiffusion(nn.Module):
    def __init__(self, temporal_dim: int):
        super(UNET_MnistDiffusion, self).__init__()
        self.temporal_input_dim = temporal_dim
        self.temporal_emb_dim = 32

        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.temporal_input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.Tanh()
        )

        self.conv0 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3)),
                                   nn.LeakyReLU())
        # [32, 32, 32]

        # Add temporal here
        self.down1 = DownSampleLayer(32 + self.temporal_emb_dim, [64, 16, 16])
        self.down2 = DownSampleLayer(64, [128, 8, 8])
        self.down3 = DownSampleLayer(128, [256, 4, 4])

        self.up3 = UpSampleLayer(256, [128, 8, 8])
        self.up2 = UpSampleLayer(128, [64, 16, 16])
        self.up1 = UpSampleLayer(64, [32, 32, 32])

        self.highway0 = nn.Sequential(
            nn.Conv2d(32 + self.temporal_emb_dim, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()
        )

        self.highway1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()
        )

        self.highway2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()
        )

        self.conv_out = nn.Sequential(
            SimpleResidualLayer(lambda: nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.LeakyReLU())),
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh(),
            LambdaLayer(lambda x: 2 * x[..., 1:-1, 1:-1])
        )


    def forward(self, xs: torch.Tensor, temporal_embedding: torch.Tensor):
        t_emb = self.temporal_encoder(temporal_embedding)[..., None, None]

        d0 = self.conv0(xs)

        d0 = torch.cat([d0, torch.broadcast_to(t_emb, [-1, self.temporal_emb_dim, 32, 32])], dim=1)
        # append t_emb to channel

        d1, h1 = self.down1(d0)  # [64, 16, 16]
        d2, h2 = self.down2(d1)  # [128, 8, 8]
        d3, h3 = self.down3(d2)

        e = d3

        u2 = self.up3(e)  # [128, 8, 8]
        u2 = u2 + self.highway2(h3)
        u1 = self.up2(u2)  # [64, 16, 16]
        u1 = u1 + self.highway1(h2)
        u0 = self.up1(u1)  # [128, 32, 32]
        u0 = u0 + self.highway0(h1)

        ys = self.conv_out(u0)
        return ys



class TemporalEmbeddingModel(nn.Module):
    def __init__(self, noise_scales):
        super(TemporalEmbeddingModel, self).__init__()
        self.noise_scales = nn.Parameter(torch.tensor(noise_scales, dtype=torch.float32))
        self.noise_scales.requires_grad = False

        self.embedding = nn.Parameter(torch.stack(
            [torch.sin(torch.arange(len(noise_scales)) * 1/s) for s in torch.arange(len(noise_scales) // 8, len(noise_scales) // 2 + 1)],
            dim=1))  #  [T, 100]
        self.embedding.requires_grad = False

    def forward(self, steps):
        return self.embedding[steps]


class NoiseEstimator(nn.Module):
    def __init__(self, noise_scales: np.ndarray):
        super(NoiseEstimator, self).__init__()
        temporal_dim = len(noise_scales) // 2 - len(noise_scales) // 8 + 1
        self.temporal_embedding_model = TemporalEmbeddingModel(noise_scales)
        self.model = UNET_MnistDiffusion(temporal_dim)

    def forward(self, xs: torch.Tensor, steps: torch.Tensor):
        t_emb = self.temporal_embedding_model(steps)
        xs = xs.view(-1, 1, 28, 28)
        return self.model(xs, t_emb).reshape(-1, 28 * 28)


noise_estimator = NoiseEstimator(noise_scales)
ddpm = DDPM(noise_scales, noise_estimator)

num_trainable_params = sum([p.numel() for p in ddpm.parameters()])
print(f"Number of parameters: {num_trainable_params}")

Convert.model_to_device(ddpm)


train_set, test_set = Mnist.get(flatten_x=True)
train_loader = DataLoader(train_set, 128)

if __name__ == '__main__':
    n_epoches = 40

    show_image(Convert.to_numpy(noise_estimator.temporal_embedding_model.embedding))

    saved_model_name = "./output/ddpm_mnist1.pth"

    def train_model():
        for i in range(n_epoches):
            losses = []
            for xs, _ in tqdm(train_loader):
                xs = Convert.to_tensor(xs)
                xs = xs * 2 - 1
                steps = torch.randint(0, len(noise_scales), [xs.shape[0]])
                loss = ddpm.train_one_batch(xs, steps)
                losses.append(loss)
            print(f"AVG train loss at epoch {i}: {np.mean(losses):.4f}")

            ys = ddpm.sample(Convert.to_tensor(torch.normal(0, 1, [16, 784])))
            ys = torch.clip(ys, 0, 1)
            # ys = xs[:16]
            ys = Convert.to_numpy(ys).reshape([4, 4, 28, 28])
            show_image(ys / 2 + 0.5, n_batch_dims=2)

        Path("./output").mkdir(exist_ok=True)
        torch.save(ddpm.state_dict(), saved_model_name)



    try:
        ddpm.load_state_dict(torch.load(saved_model_name))
    except FileNotFoundError:
        print("No saved model found, begin to train...")
        train_model()

    def test_noisy_forward():
        """
        Adding noise to the sample
        :return:
        """
        img = Convert.to_tensor(train_set[0][0][None, ...])  # Adding the batch dimension
        img_10 = ddpm.noisy_forward(img, [10])[0]
        img_20 = ddpm.noisy_forward(img, [20])[0]
        img_40 = ddpm.noisy_forward(img, [40])[0]
        img_80 = ddpm.noisy_forward(img, [80])[0]

        imgs_np = Convert.to_numpy([img, img_10, img_20, img_40, img_80])
        imgs_np = np.concatenate(imgs_np).reshape([-1, 28, 28]) / 2 + 0.5
        show_image(imgs_np, 1)

    def test_one_step_reconstruct():
        """
        One-step reconstruction: predict the noise and directly reconstruct the original sample.
        Not the same as the step-by-step manner during sampling
        :return:
        """
        img = Convert.to_tensor(train_set[0][0][None, ...])  # Adding the batch dimension

        def reconstruct_at(step: int):
            step = torch.tensor([step])
            noisy_img = ddpm.noisy_forward(img, step)[0]
            predicted_noise = ddpm.noise_estimator(noisy_img, step)
            recovered_img = noisy_img - (1 - ddpm.accumulated_scales[step]) * predicted_noise
            recovered_img = recovered_img / ddpm.accumulated_scales[step]
            return recovered_img

        imgs_np = Convert.to_numpy([img,
                                 reconstruct_at(10),
                                 reconstruct_at(20),
                                 reconstruct_at(40),
                                 reconstruct_at(80)])
        imgs_np = np.concatenate(imgs_np, axis=0).reshape([-1, 28, 28]) / 2 + 0.5
        show_image(imgs_np, 1)

    def test_sampling_steps():
        ys = Convert.to_tensor(torch.normal(0, 1, [1, 784]))

        records = [ys]
        with torch.no_grad():
            for i in range(len(ddpm.noise_scales) - 1, -1, -1):
                step_tensor = torch.ones((ys.shape[0],), device=ys.device, dtype=torch.long) * i

                if i != 0:
                    posterion_std = torch.sqrt(ddpm.noise_scales[i] * (1 - ddpm.accumulated_scales[i - 1]) / (1 - ddpm.accumulated_scales[i]))
                else:
                    posterion_std = 0

                ys = (1 / torch.sqrt(1 - ddpm.noise_scales[i])) * \
                     (ys - ddpm.noise_scales[i]/torch.sqrt(1 - ddpm.accumulated_scales[i]) * ddpm.noise_estimator(ys, step_tensor))\
                     + posterion_std * torch.normal(0, 1, ys.shape, device=ys.device)
                if i in [250, 200, 150, 100, 50, 0]:
                    records.append(ys)

        imgs_np = Convert.to_numpy(records)
        imgs_np = np.concatenate(imgs_np, axis=0).reshape([-1, 28, 28]) / 2 + 0.5
        imgs_np = np.clip(imgs_np, 0, 1)
        show_image(imgs_np, 1)



#    test_noisy_forward()
#    test_one_step_reconstruct()
    for i in range(10):
        test_sampling_steps()
