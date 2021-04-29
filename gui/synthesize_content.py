import json
from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar

from gui.synthesize_display import SynthesizeDisplay
from gui.synthesize_option_list import SynthesizeOptionList


class SynthesizeContent(QWidget):
    started_synthesizing = pyqtSignal()
    finished_synthesizing = pyqtSignal()

    def __init__(
        self, parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.vertical_layout = QVBoxLayout(self)
        self.upper_layout = QHBoxLayout()
        self.synthesize_option_list = SynthesizeOptionList(self)
        self.synthesize_option_list.started_synthesizing.connect(
            pyqtSlot()(lambda: self.started_synthesizing.emit())
        )
        self.synthesize_option_list.finished_synthesizing.connect(
            pyqtSlot()(lambda: self.finished_synthesizing.emit())
        )
        self.synthesize_display = SynthesizeDisplay(self)

        self.upper_layout.addWidget(self.synthesize_option_list)
        self.upper_layout.addWidget(self.synthesize_display)

        self.vertical_layout.addLayout(self.upper_layout)

        self.synthesize_option_list.bar_advanced.connect(self.update_bar)
        self.synthesize_option_list.finished_synthesizing.connect(
            self.populate_image_display
        )
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.vertical_layout.addWidget(self.progress_bar)

        self.setLayout(self.vertical_layout)

    @pyqtSlot(int)
    def update_bar(self, value: int):
        self.progress_bar.setValue(value)

    @pyqtSlot()
    def populate_image_display(self):
        self.progress_bar.setValue(100)

        with open("images.json") as f:
            self.synthesize_display.load_images(json.load(f))
