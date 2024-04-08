import numpy as np
import cv2


class ConnectedComponents:

    def __init__(self, images):
        self.images = images

    def count_components(self, image, labels_returned=True):
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            image, connectivity=8)

        if labels_returned:
            return num_labels - 1, labels
        else:
            return num_labels - 1, None

    def get_component_images(self):
        component_images = dict()
        num_components = []

        for i in range(len(self.images)):
            num_labels, labels = self.count_components(self.images[i])
            num_components.append(num_labels)

            component_image = np.zeros_like(self.images[i])

            for label in range(1, num_labels + 1):
                component_mask = (labels == label).astype(np.uint8) * 255
                component_image = cv2.add(component_image, component_mask)

            component_images[i] = component_image

        return component_images, num_components
