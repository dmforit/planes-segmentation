from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2


def main():
    im1 = Image.open('./media/images/example_1.jpg')
    im2 = Image.open('./media/images/example_2.jpg')
    im3 = Image.open('./media/images/example_3.jpg')

    raw_images = {0: np.array(im1),
                  1: np.array(im2),
                  2: np.array(im3)}

    images = {0: cv2.cvtColor(raw_images[0], cv2.COLOR_RGB2HSV),
              1: cv2.cvtColor(raw_images[1], cv2.COLOR_RGB2HSV),
              2: cv2.cvtColor(raw_images[2], cv2.COLOR_RGB2HSV)}

    grays = {0: images[0][..., 0],
             1: images[1][..., 0],
             2: images[2][..., 0]}

    for i in range(3):
        grays[i][(grays[i] < 90) | (grays[i] > 127)] = 0
        grays[i][grays[i] != 0] = 255

    kirsch_kernels = np.array([[[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
                               [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
                               [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
                               [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
                               [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
                               [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
                               [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
                               [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]])

    kirsch_images = {0: [], 1: [], 2: []}

    for i in range(kirsch_kernels.shape[0]):
        kernel = kirsch_kernels[i]
        for j in range(len(kirsch_images)):
            kirsch_images[j].append(signal.convolve2d(grays[j], kernel))

    for i in range(len(kirsch_images)):
        kirsch_images[i] = np.max(kirsch_images[i], axis=0)
        kirsch_images[i][kirsch_images[i] > 255] = 255
        kirsch_images[i] = kirsch_images[i].astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    for i in range(len(kirsch_images)):
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            kirsch_images[i], connectivity=8)

        component_image = np.zeros_like(kirsch_images[i])

        # Draw each component on the image
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            component_image = cv2.add(component_image, component_mask)

        ax[i].imshow(component_image, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f'Число самолётов: {num_labels - 1}')

    plt.show()


if __name__ == '__main__':
    main()
