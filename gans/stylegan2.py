from typing import Optional
import pickle
from urllib import request
import os

import torch
import numpy as np

from gans.base import BaseGAN


class StyleGAN2(BaseGAN):
    """
    StyleGAN2 wrapper class
    Attributes:
        device - device on which StyleGAN2 is loaded
        G - generator
    """
    def __init__(self, device: torch.device, weights_path: Optional[str] = None) -> None:
        """
        Initialize StyleGAN2
        :param device: device on which StyleGAN2 is loaded
        :param weights_path: path with weights
        """
        super(StyleGAN2, self).__init__()
        self.device = device
        if weights_path is None:
            weights_path = 'ffhq.pkl'
            if not os.path.exists(weights_path):
                print("Weights of StyleGAN2 are not specified and not found, downloading...")
                remote_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
                request.urlretrieve(remote_url, weights_path)
                print("Weights downloaded")
        with open(weights_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)
        self.z_dim = self.G.z_dim
        print("Weights are loaded successfully!")

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        rng = np.random.RandomState(seed)
        z = torch.from_numpy(rng.standard_normal(self.z_dim * batch_size).reshape(batch_size,
                                                                                  self.z_dim)).float().to(self.device)
        with torch.no_grad():
            return self.G.mapping(z, None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z, None)
