import random

import PIL.Image
import cv2
import numpy as np
import skimage.restoration
import skimage.util
from matplotlib import pyplot as plt, colors
from skimage import img_as_ubyte

import facelab.imageset


class Image(object):
    """
    :argument
        name: image name
        npimg: image numppy data
        bboxs: boxs with shape [(x, y, width, height), ...]
        landmarks: landmarks with shape [(x1, y1, x2, y2, ....), (...), ...]
    """
    def __init__(self, npimg, name=''):
        self.name = str(name)[:10] + '..' if len(name) > 20 else str(name)
        self.npimg = npimg
        self.bboxs = None
        self.landmarks = None

    @classmethod
    def from_example(cls, example, dataset):
        name = example.get(dataset.search_feature)
        npimg = example.get(dataset.image_feature)
        bbox = example.get(dataset.bbox_feature)
        landmark = example.get(dataset.landmark_feature)
        return cls(name = name, npimg = npimg).with_bbox(bbox).with_landmarks(landmark)

    def __str__(self):
        if self.name:
            return '<Image {} {}>'.format(self.name, self.size)
        else:
            return '<Image {}>'.format(self.size)

    def add_plot(self, with_bbox=False, with_lanmark=False, fontsize=6):
        plt.title(self, fontsize = fontsize)
        img = self
        if with_bbox:
            for x, y, w, h in self.bboxs:
                img = img.draw_rect(x, y, w, h)
        if with_lanmark:
            for landmark in self.landmarks:
                for p in range(0, len(landmark)-1, 2):
                    img = img.draw_point(landmark[p], landmark[p+1])
        plt.imshow(img.npimg, cmap = 'bone')

    def show(self, with_bbox=False, with_landmark=False, fontsize=6):
        self.add_plot(with_bbox, with_landmark, fontsize)
        plt.axis('off')
        plt.show()

    def with_landmarks(self, landmarks):
        self.landmarks = landmarks
        return self

    def with_bbox(self, bboxs):
        self.bboxs = bboxs
        return self

    def with_name(self, name):
        self.name = name
        return self

    @staticmethod
    def color2rgb(color):
        return tuple(map(lambda c: 255 * c, colors.to_rgb(color)))

    def convert(self, cmap='COLOR_BGR2RGB'):
        return Image(cv2.cvtColor(self.npimg, getattr(cv2, cmap)))

    def laplacian(self):
        return Image(cv2.Laplacian(self.npimg, cv2.CV_8U))

    def draw_text(self, text, posx, posy, color):
        rgb = tuple(map(lambda x: 255 * x, colors.to_rgb(color)))
        return Image(cv2.putText(self.npimg, str(text), (posx, posy),
                                 fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = rgb))

    def draw_rect(self, posx, posy, width, height, color='blue', thickness=2, **kwargs):
        rgb = self.color2rgb(color)
        return Image(cv2.rectangle(self.npimg, (posx, posy), (posx + width, posy + height),
                                   color = rgb, thickness = thickness, **kwargs))

    def draw_point(self, x, y, size=1, color='red', thickness=-1):
        rgb = self.color2rgb(color)
        npimg = cv2.circle(self.npimg, (x, y), radius = size, color = rgb, thickness = thickness)
        return Image(npimg)

    def add_noise(self, mode='gaussian'):
        return Image(img_as_ubyte(skimage.util.random_noise(self.npimg, mode = mode)))

    def pad_border(self, top, bottom, left, right, mode=cv2.BORDER_REPLICATE):
        return Image(cv2.copyMakeBorder(self.npimg, top, bottom, left, right, mode))

    def blur(self, ksize, sigmax, sigmay=None, mode='gaussian'):
        if mode == 'gaussian':
            if not sigmay:
                sigmay = sigmax
            return Image(cv2.GaussianBlur(self.npimg, ksize, sigmax, sigmaY = sigmay))

    def var(self):
        return self.npimg.var()

    def crop(self, top, bottom, left, right):
        pil_img = PIL.Image.fromarray(self.npimg).crop((left, top, right, bottom))
        return Image(np.asarray(pil_img))

    def save(self, filename):
        cv2.imwrite(filename, self.convert('COLOR_RGB2BGR').npimg)

    def resize(self, width, height, mode='INTER_LINEAR'):
        return Image(cv2.resize(self.npimg, (width, height), interpolation = getattr(cv2, mode)))

    @property
    def size(self):
        return '{}x{}'.format(self.npimg.shape[1], self.npimg.shape[0])

    @property
    def width(self):
        return self.npimg.shape[1]

    @property
    def height(self):
        return self.npimg.shape[0]

    def denoise(self, mode='bilateral', *args, **kwargs):
        """https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle"""
        return Image(img_as_ubyte(getattr(skimage.restoration, 'denoise_{}'.format(mode))
                                  (self.npimg, *args, **kwargs)))

    def hist(self):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(self.npimg)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.hist(self.npimg.ravel(), 256)
        plt.show()

    def equalized_hist(self):
        """https://opencv-python-tutroals.readthedocs.io/en/latest/
        py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
        https://hypjudy.github.io/2017/03/19/dip-histogram-equalization/"""
        img_yuv = cv2.cvtColor(self.npimg, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return Image(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))

    def sample_noise(self, n=1):
        candicate_method = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
        imset = facelab.imageset.ImageSet()
        for _ in range(n):
            mode = random.sample(candicate_method, 1)[0]
            imset.add(self.add_noise(mode = mode))
        return imset
