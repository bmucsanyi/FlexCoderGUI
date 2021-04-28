from typing import Optional

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar


class TabContent(QWidget):
    def __init__(
        self,
        parameter_widget: QWidget,
        output_widget: QWidget,
        parent: Optional[QWidget] = None,
        has_bar: bool = False,
    ):
        super().__init__(parent)

        self.vertical_layout = QVBoxLayout(self)
        self.upper_layout = QHBoxLayout()
        self.parameter_widget = parameter_widget

        self.output_widget = output_widget

        self.upper_layout.addWidget(self.parameter_widget)
        self.upper_layout.addWidget(self.output_widget)

        self.vertical_layout.addLayout(self.upper_layout)

        if has_bar:
            self.parameter_widget.bar_advanced.connect(self.update_bar)
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setValue(0)
            self.vertical_layout.addWidget(self.progress_bar)

        self.setLayout(self.vertical_layout)

        # self.show()

    def add_to_lower(self, widget: QWidget):
        self.lower_layout.addWidget(widget)

    def update_bar(self, value: int):
        self.progress_bar.setValue(value)
