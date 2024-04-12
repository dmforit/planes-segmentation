import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from Segmentation import Segmentation
from ConnectedComponents import ConnectedComponents
import numpy as np


class CenteredButtonsApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Сегментация")

        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.processed_image = QLabel()
        self.image_appeared = False
        self.image = None
        self.filename = None

        self.result_number = QLabel()
        self.result_number.setText('')

        self.import_button = QPushButton("Выбрать изображение")
        self.import_button.clicked.connect(self.import_image)
        self.import_button.setFixedSize(180, 40)

        self.crop_button = QPushButton("Выделить случайный кадр")
        self.crop_button.clicked.connect(self.random_crop)
        self.crop_button.setFixedSize(180, 40)

        self.process_button = QPushButton("Обработать изображение")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setFixedSize(180, 40)

        self.buttons_layout = QHBoxLayout()

        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.buttons_layout.addWidget(self.process_button,
                                      alignment=Qt.AlignmentFlag.AlignCenter)
        self.buttons_layout.addWidget(self.crop_button,
                                      alignment=Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(self.result_number,
                              alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addStretch()
        self.layout.addWidget(self.import_button,
                              alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addStretch()

        self.images_layout = QHBoxLayout()
        self.images_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.images_layout.addWidget(self.image_label)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def random_crop(self):
        if self.filename is None:
            return

        image = Segmentation([]).read_images([self.filename])[0]

        height_size = np.random.randint(1, image.shape[0] + 1)
        width_size = np.random.randint(1, image.shape[1] + 1)

        max_width = image.shape[1] - width_size
        max_height = image.shape[0] - height_size

        y = np.random.randint(0, max_height + 1)
        x = np.random.randint(0, max_width + 1)

        self.image = np.array(image[y:(y +
                                       height_size), x:(x + width_size), :])

        changed_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                               self.image.shape[1] * 3, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(changed_image).scaled(700, 600)

        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image.hide()
        self.result_number.setText('')

    def process_image(self):
        if self.filename is None:
            return
        segmentation = Segmentation([])
        segmentation.gray_images = segmentation.hsv2gray(
            segmentation.rgb2hsv({0: self.image}))
        segmented_images = segmentation.get_segmentation()

        connected_components = ConnectedComponents(segmented_images)
        images, planes_numbers = connected_components.get_component_images()

        self.result_number.setText(
            f'Число самолётов на изображении ~ {planes_numbers[0]}')

        image = QImage(images[0].data, images[0].shape[1], images[0].shape[0],
                       images[0].shape[1], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image).scaled(700, 600)

        self.processed_image.setPixmap(pixmap)
        self.processed_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.images_layout.addWidget(self.processed_image)
        self.processed_image.show()

    def import_image(self):
        self.filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if self.filename:
            self.image = Segmentation([]).read_images([self.filename])[0]
            pixmap = QPixmap(self.filename).scaled(700, 600)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if self.layout.count() == 6:
                self.result_number.setText('')

            if not self.image_appeared:
                self.layout.insertLayout(0, self.images_layout)
                self.layout.insertLayout(
                    self.layout.count() - 1, self.buttons_layout)
                self.image_appeared = True
            elif self.images_layout.count() == 2:
                self.images_layout.removeItem(self.images_layout.itemAt(1))
                self.processed_image.hide()


def main():
    app = QApplication(sys.argv)
    window = CenteredButtonsApp()
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
