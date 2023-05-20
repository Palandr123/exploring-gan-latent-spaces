from typing import Optional
from models import Generator

import torch
import numpy as np

from gans.base import BaseGAN


class FastGAN(BaseGAN):
    """
    StyleGAN2 wrapper class
    Attributes:
        device - device on which StyleGAN2 is loaded
        G - generator
    """
    def __init__(self, device: torch.device, weights_path: Optional[str] = None) -> None:
        """
        Initialize FastGAN
        :param device: device on which FastGAN is loaded
        :param weights_path: path with weights
        """
        super(FastGAN, self).__init__()
        self.device = device
        self.G = Generator(nz=256, im_size=512).to(device)
        checkpoint = torch.load(weights_path, map_location=lambda a, b: a)
        checkpoint['g_ema'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        self.G.load_state_dict(checkpoint['g_ema'])
        self.z_dim = 256
        print("Weights are loaded successfully!")

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        rng = np.random.RandomState(seed)
        z = torch.from_numpy(rng.standard_normal(self.z_dim * batch_size).reshape(batch_size,
                                                                                  self.z_dim)).float().to(self.device)
        return z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z)[0]
