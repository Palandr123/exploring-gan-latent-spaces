from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGAN(ABC, nn.Module):
    def __init__(self) -> None:
        super(BaseGAN, self).__init__()

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        forward method for GAN
        :param z: latent vector, torch.tensor of shape (N, z_dim)
        :return: generated images, torch.tensor of shape (N, img_height, img_width)
        """
        pass

    @abstractmethod
    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        """
        sample a batch of latent vectors
        :param batch_size: number of latent vectors to sample
        :param seed: seed with which latent vectors are samples
        :return: batch of latent vectors
        """
        pass
