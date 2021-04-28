from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from gui.train_display import TrainDisplay
from gui.train_option_list import TrainOptionList


class TrainContent(QWidget):
    finished_training = pyqtSignal()

    def __init__(
        self, parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.upper_layout = QHBoxLayout(self)
        self.train_option_list = TrainOptionList(self)
        self.train_option_list.can_write.connect(self.write_string)

        self.train_display = TrainDisplay(self)

        self.upper_layout.addWidget(self.train_option_list)
        self.upper_layout.addWidget(self.train_display)

        self.setLayout(self.upper_layout)

    @pyqtSlot(str)
    def write_string(self, value: str):
        self.train_display.write_new_text(value)
