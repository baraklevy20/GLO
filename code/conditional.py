import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class _netG(nn.Module):
    def __init__(self, number_of_latents, number_of_filters, number_of_channels):
        super().__init__()
        self.dcgan = nn.Sequential(
            nn.ConvTranspose2d(number_of_latents, number_of_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(number_of_filters * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(number_of_filters * 8, number_of_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_filters * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(number_of_filters * 4, number_of_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_filters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(number_of_filters * 2, number_of_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(number_of_filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(number_of_filters, number_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.dcgan(input)
