from typing import Optional

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class TabBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.horizontal_layout = QHBoxLayout(self)
        self.generating_button = QPushButton("Data generating", self)
        self.generating_button.setFont(QFont("Roboto", 20))
        self.generating_button.setStyleSheet("""
            border-bottom-left-radius:0px;
            border-bottom-right-radius:0px;
        """)

        self.training_button = QPushButton("Training", self)
        self.training_button.setFont(QFont("Roboto", 20))
        self.training_button.setStyleSheet("""
            border-bottom-left-radius:0px;
            border-bottom-right-radius:0px;
        """)

        self.synthesis_button = QPushButton("Synthesizing", self)
        self.synthesis_button.setFont(QFont("Roboto", 20))
        self.synthesis_button.setStyleSheet("""
            border-bottom-left-radius:0px;
            border-bottom-right-radius:0px;
        """)

        self.horizontal_layout.addWidget(self.generating_button)
        self.horizontal_layout.addWidget(self.training_button)
        self.horizontal_layout.addWidget(self.synthesis_button)

        self.setLayout(self.horizontal_layout)

        # self.show()
        self.generating_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.synthesis_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.horizontal_layout.setSpacing(0)
        self.setFixedHeight(100)
