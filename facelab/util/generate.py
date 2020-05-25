# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.io import imsave
import cv2
from skimage import img_as_ubyte

im = cv2.imread('../data/clear-1.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = random_noise(im, mode='s&p')
plt.imshow(im)
plt.show()
imsave('../data/noise-s&p-1.jpg', img_as_ubyte(im))

