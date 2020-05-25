import math

import facelab.image

import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageSet(object):
    """imgpaths -> npimgs -> images"""

    def __init__(self, imgpaths=None, npimgs=None, images=None):
        self.images = images if images else []
        self.npimgs = npimgs if npimgs else np.array([])
        self.imgpaths = imgpaths if imgpaths else []

    @classmethod
    def from_imgpaths(cls, imgpaths):
        pass

    @classmethod
    def from_npimgs(cls, npimgs):
        return cls(npimgs = npimgs)

    @classmethod
    def from_images(cls, images):
        return cls(images = images)

    @property
    def size(self):
        return len(self.images)

    def __add__(self, other):
        if isinstance(other, ImageSet):
            self.images += other.images
        elif isinstance(other, facelab.image.Image):
            self.images.append(other)
        return self

    def load(self):
        for imgpath in self.imgpaths:
            img = facelab.image.Image(cv2.imread(imgpath)).convert()
            self.add(img)

    def add(self, image):
        if isinstance(image, facelab.image.Image):
            self.images.append(image)
            # np.append(self.npimgs, img.npimg, axis=0)

    def loc(self, idx):
        return self.images[idx]

    def show(self, limit=5, column=6, figsize=None, dpi=None, cmap=None, with_bbox=False, with_landmark=False):
        # calc the number of show table row
        if self.size < column:
            column = self.size
        rows = math.ceil(self.size / column)

        # define figure and subplot on it
        plt.figure(figsize=figsize, dpi=dpi)
        shows = self.images[:limit]
        for idx, img in enumerate(shows):
            plt.subplot(rows, column, idx+1)
            plt.axis('off')
            if cmap:
                img = img.convert(cmap = cmap)
            img.add_plot(with_bbox, with_landmark)
        plt.show()



