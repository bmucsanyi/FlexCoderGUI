from typing import Optional

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy


class TabButton(QPushButton):
    pass


class GeneratingButton(TabButton):
    pass


class TrainingButton(TabButton):
    pass


class SynthesizeButton(TabButton):
    pass


class TabBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.horizontal_layout = QHBoxLayout(self)
        self.generating_button = GeneratingButton("Data Generation", self)
        self.generating_button.setFont(QFont("Roboto", 20))

        self.training_button = TrainingButton("Training", self)
        self.training_button.setFont(QFont("Roboto", 20))

        self.synthesis_button = SynthesizeButton("Synthesizing", self)
        self.synthesis_button.setFont(QFont("Roboto", 20))

        self.horizontal_layout.addWidget(self.generating_button)
        self.horizontal_layout.addWidget(self.training_button)
        self.horizontal_layout.addWidget(self.synthesis_button)

        self.setLayout(self.horizontal_layout)

        self.generating_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.training_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.synthesis_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.horizontal_layout.setSpacing(0)
        self.setFixedHeight(100)
