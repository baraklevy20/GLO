import numpy as np
from scipy import signal

def compute_kernel(kernel_size):
    # Get the kernel by applying a convolution of [1 1] on itself [kernel_size - 1] times
    kernel = np.array([[1]]).astype(np.float64)
    for i in range(kernel_size - 1):
        kernel = signal.convolve2d(kernel, np.array([[1, 1]]))

    # To normalize it, we need to divide by the total sum of the kernel.
    # The sum of the n'th row of the pascal triangle is 2^(n-1)
    return kernel / (2 ** (kernel_size - 1))


def build_gaussian_pyramid(im, max_levels, filter_size):
    kernel = compute_kernel(filter_size).astype(np.float64)
    pyramids = []

    # Go through each level
    for i in range(max_levels):
        # Add the current image to the pyramid
        pyramids.append(im)

        # Reduce the image
        im = signal.convolve2d(im, kernel, mode='same', boundary='symm')
        im = signal.convolve2d(im, kernel.T, mode='same', boundary='symm')
        im = im[::2, ::2]

        # Stop if the image gets too small
        if im.shape[0] < 16 or im.shape[1] < 16:
            break

    return pyramids, kernel


def build_laplacian_pyramid(im, max_levels, filter_size):
    # Compute the Gaussian pyramids
    gaussians, kernel = build_gaussian_pyramid(im, max_levels, filter_size)

    # Use it to get the Laplacian pyramids
    laplacians = list(((gaussians[i] - expand(gaussians[i + 1], kernel)) for i in range(len(gaussians) - 1)))

    # The last pyramid is the last Gaussian pyramid
    laplacians.append(gaussians[len(gaussians) - 1])
    return laplacians, kernel


def expand(im, kernel):
    # Enlarge the image
    new_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]))
    new_im[:: 2, :: 2] = im

    # Blur it
    new_im = signal.convolve2d(new_im, 2 * kernel, mode='same', boundary='symm')
    new_im = signal.convolve2d(new_im, 2 * kernel.T, mode='same', boundary='symm')

    return new_im


def laplacian_loss(im1, im2, max_level):
    lap1 = build_laplacian_pyramid(im1, max_level, 5)
    lap2 = build_laplacian_pyramid(im2, max_level, 5)

    sum = 0
    for j in range(max_level):
        sum += 2 ** (- 2 * j) * abs(lap1[j] - lap2[j])

    return sum