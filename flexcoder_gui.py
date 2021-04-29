import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication

from gui.data_content import DataContent
from gui.synthesize_content import SynthesizeContent
from gui.tab_bar import TabBar
from gui.train_content import TrainContent


class BaseContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.vertical_layout = QVBoxLayout(self)
        self.tab_bar = TabBar(self)
        self.tab_bar.generating_button.clicked.connect(self.switch_to_generate)
        self.tab_bar.training_button.clicked.connect(self.switch_to_train)
        self.tab_bar.synthesis_button.clicked.connect(self.switch_to_synthesis)

        self.data_content = DataContent(self)
        self.data_content.started_generating.connect(self.disable_for_generating)
        self.data_content.finished_generating.connect(self.enable_for_generating)
        self.train_content = TrainContent(self)
        self.train_content.started_training.connect(self.disable_for_training)
        self.train_content.finished_training.connect(self.enable_for_training)
        self.synthesize_content = SynthesizeContent(self)
        self.synthesize_content.started_synthesizing.connect(self.disable_for_synthesis)
        self.synthesize_content.finished_synthesizing.connect(self.enable_for_synthesis)

        self.vertical_layout.addWidget(self.tab_bar)
        self.vertical_layout.addWidget(self.data_content)
        self.vertical_layout.addWidget(self.train_content)
        self.vertical_layout.addWidget(self.synthesize_content)

        self.switch_to_generate()

        self.setLayout(self.vertical_layout)
        self.setWindowTitle("FlexCoder - Program Synthesis Tool")

        self.setFixedSize(1000, 600)

        self.setWindowIcon(QIcon("gui/images/kiwi.svg"))

        self.show()

    def switch_to_generate(self):
        self.switch(self.data_content, "GeneratingButton")

    def switch_to_train(self):
        self.switch(self.train_content, "TrainingButton")

    def switch_to_synthesis(self):
        self.switch(self.synthesize_content, "SynthesizeButton")

    def switch(self, content, class_str):
        other_contents = [
            self.data_content,
            self.train_content,
            self.synthesize_content,
        ]
        other_contents = list(filter(lambda x: x is not content, other_contents))

        content.setHidden(False)
        for other in other_contents:
            other.setHidden(True)

        self.set_normal_style_sheet(class_str)

    def set_enabled_tab_bar(self, is_enabled):
        self.tab_bar.generating_button.setEnabled(is_enabled)
        self.tab_bar.training_button.setEnabled(is_enabled)
        self.tab_bar.synthesis_button.setEnabled(is_enabled)

    @pyqtSlot()
    def disable_for_generating(self):
        self.set_enabled_tab_bar(False)

        self.set_disabled_style_sheet(
            "GeneratingButton", "TrainingButton", "SynthesizeButton"
        )

    @pyqtSlot()
    def disable_for_training(self):
        self.set_enabled_tab_bar(False)

        self.set_disabled_style_sheet(
            "TrainingButton", "GeneratingButton", "SynthesizeButton"
        )

    @pyqtSlot()
    def disable_for_synthesis(self):
        self.set_enabled_tab_bar(False)

        self.set_disabled_style_sheet(
            "SynthesizeButton", "TrainingButton", "GeneratingButton"
        )

    @pyqtSlot()
    def enable_for_generating(self):
        self.set_enabled_tab_bar(True)

        self.set_normal_style_sheet("GeneratingButton")

    @pyqtSlot()
    def enable_for_training(self):
        self.set_enabled_tab_bar(True)

        self.set_normal_style_sheet("TrainingButton")

    @pyqtSlot()
    def enable_for_synthesis(self):
        self.set_enabled_tab_bar(True)

        self.set_normal_style_sheet("SynthesizeButton")

    def set_normal_style_sheet(self, cls_1):
        self.setStyleSheet(
            f"""
            QWidget {{
                background: #303030;
                color: white;
            }}
            QPushButton {{
                padding:0.3em 1.2em;
                margin:0 0.3em 0.3em 0;
                border-radius:10px;
                text-decoration:none;
                font-family:'Roboto',sans-serif;
                font-weight:300;
                color:#FFFFFF;
                background-color:#34655b;
            }}
            QPushButton:hover {{
                background-color:#40ab5b;
            }}
            TabButton {{
                border-bottom-left-radius:0px;
                border-bottom-right-radius:0px;
            }}
            QLineEdit {{
                background-color: white;
                color: black;
            }}
            {cls_1}:!hover {{
                background-color:#40855b;
            }}
            QSlider::handle:horizontal {{
                background-color:#34655b;
            }}
            QSlider::handle:horizontal:hover {{
                background-color:#40ab5b;
            }}
            """
        )

    def set_disabled_style_sheet(self, cls_1, cls_2, cls_3):
        self.setStyleSheet(
            f"""
            QWidget {{
                background: #303030;
                color: white;
            }}
            QPushButton {{
                padding:0.3em 1.2em;
                margin:0 0.3em 0.3em 0;
                border-radius:10px;
                text-decoration:none;
                font-family:'Roboto',sans-serif;
                font-weight:300;
                color:#FFFFFF;
                background-color:#34655b;
            }}
            QPushButton:hover {{
                background-color:#40ab5b;
            }}
            TabButton {{
                border-bottom-left-radius:0px;
                border-bottom-right-radius:0px;
            }}
            QLineEdit {{
                background-color: white;
                color: black;
            }}
            {cls_1}:!hover {{
                background-color:#40855b;
            }}
            {cls_2}:!hover {{
                background-color:#a1a1a1
            }}
            {cls_3}:!hover {{
                background-color:#a1a1a1
            }}
            QSlider::handle:horizontal {{
                background-color:#34655b;
            }}
            QSlider::handle:horizontal:hover {{
                background-color:#40ab5b;
            }}
            """
        )


app = QApplication(sys.argv)
w = BaseContainer()
app.exec_()
