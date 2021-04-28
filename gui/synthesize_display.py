from typing import Optional

from PyQt5.QtGui import QFont, QPainter
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt5.QtSvg import QSvgWidget, QSvgRenderer
from PyQt5.QtCore import Qt, QSize


class SynthesizeDisplay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.image_list = None
        self.image_index = None
        self.vertical_layout = QVBoxLayout(self)

        self.input_label = QLabel("Awaiting inputs...", self)
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedHeight(50)
        self.input_label.setFont(QFont("Roboto", 15))

        self.output_label = QLabel("Awaiting outputs...", self)
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedHeight(50)
        self.output_label.setFont(QFont("Roboto", 15))

        self.image_label = QLabel(parent=self)
        self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label = QSvgWidget()

        self.horizontal_layout = QHBoxLayout()
        self.back_button = QPushButton(parent=self)
        back_icon = QIcon("gui/images/left_arrow.png")
        self.back_button.setIcon(back_icon)
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self.previous_image)

        self.forward_button = QPushButton(parent=self)
        forward_icon = QIcon("gui/images/right_arrow.png")
        self.forward_button.setIcon(forward_icon)
        self.forward_button.setEnabled(False)
        self.forward_button.clicked.connect(self.next_image)

        # self.image_label.setScaledContents(True)
        self.image_label.setText("Awaiting results...")
        self.image_label.setFont(QFont("Roboto", 15))

        self.horizontal_layout.addWidget(self.back_button)
        self.horizontal_layout.addWidget(self.forward_button)

        self.vertical_layout.addWidget(self.input_label)
        self.vertical_layout.addWidget(self.output_label)
        self.vertical_layout.addWidget(self.image_label)
        self.vertical_layout.addLayout(self.horizontal_layout)

        self.setLayout(self.vertical_layout)
        # self.show()

    def load_images(self, image_list):
        self.image_list = image_list
        self.image_index = 0

        if len(self.image_list) > 1:
            self.back_button.setEnabled(True)
            self.forward_button.setEnabled(True)

        self.display_image()

    def display_image(self):
        input_ = self.image_list[self.image_index]["input"]
        output = self.image_list[self.image_index]["output"]
        path = self.image_list[self.image_index]["path"]

        self.input_label.setText(f"Input: {input_}")
        self.output_label.setText(f"Output: {output[0]}")
        self.image_label.setText("")
        if path is not None:
            pm = QPixmap(path)
            pm = pm.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pm)
        else:
            self.image_label.setPixmap(QPixmap(None))
            self.image_label.setText("No solution for the given problem.")

    def previous_image(self):
        self.image_index -= 1
        if self.image_index == -1:
            self.image_index = len(self.image_list) - 1
        self.display_image()

    def next_image(self):
        self.image_index += 1
        if self.image_index == len(self.image_list):
            self.image_index = 0
        self.display_image()


# app = QApplication(sys.argv)
# w = TrainDisplay()
# app.exec_()
