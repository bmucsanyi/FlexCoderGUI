from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QMessageBox


class BaseOptionList(QWidget):
    def set_up_section(self, text, minimum, maximum, step_size=1, exp=False):
        if exp:
            label = QLabel(text + f": {2 ** minimum}", self)
        else:
            label = QLabel(text + f": {minimum * step_size}", self)
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(minimum)

        if exp:
            slider.valueChanged.connect(
                pyqtSlot()(
                    lambda value: label.setText(
                        " ".join(label.text().split()[:-1]) + " " + str(2 ** value)
                    )
                )
            )
        else:
            slider.valueChanged.connect(
                pyqtSlot()(
                    lambda value: label.setText(
                        " ".join(label.text().split()[:-1])
                        + " "
                        + str(value * step_size)
                    )
                )
            )

        return label, slider

    def warn(self, text: str):
        warning_screen = QMessageBox()
        warning_screen.setFixedSize(500, 200)
        warning_screen.critical(self, "Error", text)
