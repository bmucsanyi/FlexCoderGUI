import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication

from gui.tab_bar import TabBar
from gui.data_content import DataContent
from gui.train_content import TrainContent
from gui.synthesize_content import SynthesizeContent


class BaseContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.vertical_layout = QVBoxLayout(self)
        self.tab_bar = TabBar(self)
        self.tab_bar.generating_button.clicked.connect(self.switch_to_generate)
        self.tab_bar.training_button.clicked.connect(self.switch_to_train)
        self.tab_bar.synthesis_button.clicked.connect(self.switch_to_synthesis)

        self.data_content = DataContent(self)
        self.train_content = TrainContent(self)
        self.synthesize_content = SynthesizeContent(self)

        self.vertical_layout.addWidget(self.tab_bar)
        self.vertical_layout.addWidget(self.data_content)
        self.vertical_layout.addWidget(self.train_content)
        self.vertical_layout.addWidget(self.synthesize_content)

        self.switch_to_generate()

        self.setLayout(self.vertical_layout)
        self.setWindowTitle("FlexCoder GUI")

        self.show()

    def switch_to_generate(self):
        self.data_content.setHidden(False)
        self.train_content.setHidden(True)
        self.synthesize_content.setHidden(True)

    def switch_to_train(self):
        self.data_content.setHidden(True)
        self.train_content.setHidden(False)
        self.synthesize_content.setHidden(True)

    def switch_to_synthesis(self):
        self.data_content.setHidden(True)
        self.train_content.setHidden(True)
        self.synthesize_content.setHidden(False)


app = QApplication(sys.argv)
w = BaseContainer()
app.exec_()
