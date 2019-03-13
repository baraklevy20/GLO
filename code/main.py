import conditional
import numpy as np
from matplotlib.image import imread

from laplacian_loss import laplacian_loss

generator = conditional._netG(100, 3, 3)
im1 = imread('../data/im64x64.jpg')
im2 = imread('../data/im64x64.jpg')
print(laplacian_loss(im1, im2, 3))