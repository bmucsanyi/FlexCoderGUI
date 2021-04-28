from typing import Optional

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton


class TabBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.horizontal_layout = QHBoxLayout(self)
        self.generating_button = QPushButton("Data generating", self)
        self.training_button = QPushButton("Training", self)
        self.synthesis_button = QPushButton("Synthesizing", self)

        self.horizontal_layout.addWidget(self.generating_button)
        self.horizontal_layout.addWidget(self.training_button)
        self.horizontal_layout.addWidget(self.synthesis_button)

        self.setLayout(self.horizontal_layout)

        self.show()
