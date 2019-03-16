import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_kernel(kernel_size, number_of_channels):
    grid = np.float32(np.mgrid[0:kernel_size, 0:kernel_size].T)

    # Using the real gaussian instead of an approximation we saw in image processing class
    gaussian = lambda x: np.exp(-1 / 2 * (x - kernel_size // 2) ** 2) ** 2
    kernel = np.sum(gaussian(grid), axis=2)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Repeat it for every channel
    kernel = np.tile(kernel, (number_of_channels, 1, 1))

    # Conv2D works with a kernel as a tensor
    kernel = torch.FloatTensor(kernel[:, None, :, :])

    # Convert the kernel to CPU or CUDA
    kernel = kernel.to(device)

    return kernel


def apply_kernel(img, kernel):
    n_channels, _, kernel_size, _ = kernel.shape

    # Convolve the image with the kernel. Add padding so that the size won't change
    return fnn.conv2d(img, kernel, groups=n_channels, padding=kernel_size // 2)


def build_laplacian_pyramid(im, kernel, max_levels):
    current_gaussian = im
    laplacian_pyramids = []

    for level in range(max_levels):
        # Calculate the Gaussian pyramid
        next_gaussian = apply_kernel(current_gaussian, kernel)

        # Calculate the Laplacian pyramid using the Gaussian
        laplacian_pyramids.append(current_gaussian - next_gaussian)

        # Move on to the next level of the pyramid
        current_gaussian = fnn.avg_pool2d(next_gaussian, 2)

    # The last pyramid is the last Gaussian pyramid
    laplacian_pyramids.append(current_gaussian)

    return laplacian_pyramids


class LaplacianLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.kernel = None

    def forward(self, input, target):
        # Compute the kernel once
        if self.kernel is None:
            self.kernel = compute_kernel(self.kernel_size, input.shape[1])

        # Calculate the Laplacian pyramids
        source = build_laplacian_pyramid(input, self.kernel, self.max_levels)
        target = build_laplacian_pyramid(target, self.kernel, self.max_levels)

        # Sum over the different levels
        return sum(fnn.l1_loss(s, t) for s, t in zip(source, target))
