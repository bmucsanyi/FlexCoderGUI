from typing import Optional

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar
from gui.data_display import DataDisplay
from gui.data_option_list import DataOptionList

import json


class DataContent(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.vertical_layout = QVBoxLayout(self)
        self.upper_layout = QHBoxLayout()
        self.data_option_list = DataOptionList()
        self.data_display = DataDisplay()

        self.upper_layout.addWidget(self.data_option_list)
        self.upper_layout.addWidget(self.data_display)

        self.vertical_layout.addLayout(self.upper_layout)

        self.data_option_list.bar_advanced.connect(self.update_bar)
        self.data_option_list.finished_generating.connect(self.populate_diagram)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.vertical_layout.addWidget(self.progress_bar)

        self.setLayout(self.vertical_layout)

        # self.show()

    def update_bar(self, value: int):
        self.progress_bar.setValue(value)

    def populate_diagram(self):
        self.progress_bar.setValue(100)  # TODO: why is this needed?
        with open("statistics.json") as f:
            statistics = json.load(f)
        self.data_display.populate_diagram(statistics)