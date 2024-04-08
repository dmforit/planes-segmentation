from PIL import Image
import numpy as np
from scipy import signal
import cv2


class Segmentation:

    def __init__(self, paths):
        self.paths = paths
        self.images = self.rgb2hsv(self.read_images(paths))
        self.gray_images = self.hsv2gray(self.images)

        self.low_hue = 90
        self.high_hue = 127

        self.kirsch_kernels = np.array([[[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
                                        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
                                        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
                                        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
                                        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
                                        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
                                        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
                                        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]])

    def get_segmentation(self, binarize=True, borders=True):
        result = None

        if binarize:
            result = self.binarize(self.gray_images)

        if borders:
            result = self.get_borders(
                result) if result is not None else self.get_borders1(self.gray_images)

        return result if result is not None else self.gray_images

    def binarize(self, gray_images):
        binarized = gray_images

        for i in range(len(binarized)):
            binarized[i][(binarized[i] < self.low_hue) |
                         (binarized[i] > self.high_hue)] = 0
            binarized[i][binarized[i] != 0] = 255

        return binarized

    def get_borders(self, images):
        kirsch_images = {i: [] for i in range(len(images))}

        for i in range(len(kirsch_images)):

            for kernel in self.kirsch_kernels:
                kirsch_images[i].append(signal.convolve2d(images[i], kernel))

            kirsch_images[i] = np.max(kirsch_images[i], axis=0)
            kirsch_images[i][kirsch_images[i] > 255] = 255
            kirsch_images[i] = kirsch_images[i].astype(np.uint8)

        return kirsch_images

    def read_images(self, paths):
        images = dict()

        for i in range(len(paths)):
            images[i] = np.array(Image.open(self.paths[i]))

        return images

    def rgb2hsv(self, images):
        hsv_images = dict()

        for i in range(len(images)):
            hsv_images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)

        return hsv_images

    def hsv2gray(self, hsv_images):
        grayscale_images = dict()

        for i in range(len(hsv_images)):
            grayscale_images[i] = hsv_images[i][..., 0]

        return grayscale_images
