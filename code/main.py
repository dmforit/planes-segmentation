from Segmentation import Segmentation
from ConnectedComponents import ConnectedComponents
import matplotlib.pyplot as plt


def main():
    segmentation = Segmentation(['./media/images/example_1.jpg',
                                 './media/images/example_2.jpg',
                                 './media/images/example_3.jpg'])

    segmented_images = segmentation.get_segmentation()
    connected_components = ConnectedComponents(segmented_images)
    images, planes_numbers = connected_components.get_component_images()

    _, ax = plt.subplots(len(images), 1)

    for i in range(len(images)):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f'Число самолётов: {planes_numbers[i]}')

    plt.show()


if __name__ == '__main__':
    main()
