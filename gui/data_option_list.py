from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QLineEdit,
    QSlider,
    QPushButton,
    QMessageBox,
    QFileDialog,
)

from src.generate_utils import DataWorker, Arguments


# noinspection PyUnresolvedReferences
class DataOptionList(QWidget):
    bar_advanced = pyqtSignal(int)
    finished_generating = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.thread = None
        self.worker = None
        self.filename = None

        self.vertical_layout = QVBoxLayout(self)
        self.is_test_checkbox = QCheckBox("Generate test dataset", self)
        self.num_comps_line_edit = QLineEdit(self)
        self.num_comps_line_edit.setPlaceholderText("Number of compositions")

        (
            self.num_samples_per_comp_label,
            self.num_samples_per_comp_slider,
        ) = self.set_up_section("Sample / composition", 1, 1, 10)

        self.select_path_button = QPushButton("Select path", self)
        self.select_path_button.clicked.connect(self.path_clicked)

        self.num_functions_label, self.num_functions_slider = self.set_up_section(
            "Number of functions", 1, 1, 6
        )
        self.num_io_label, self.num_io_slider = self.set_up_section(
            "Number of I/O examples", 1, 1, 4
        )
        self.num_inputs_label, self.num_inputs_slider = self.set_up_section(
            "Number of inputs", 1, 1, 5
        )
        (
            self.num_unique_inputs_label,
            self.num_unique_inputs_slider,
        ) = self.set_up_section("Number of unique inputs", 1, 1, 5)
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.start_generating)

        self.vertical_layout.addWidget(self.is_test_checkbox)
        self.vertical_layout.addWidget(self.num_comps_line_edit)
        self.vertical_layout.addWidget(self.num_samples_per_comp_label)
        self.vertical_layout.addWidget(self.num_samples_per_comp_slider)
        self.vertical_layout.addWidget(self.select_path_button)
        self.vertical_layout.addWidget(self.num_functions_label)
        self.vertical_layout.addWidget(self.num_functions_slider)
        self.vertical_layout.addWidget(self.num_io_label)
        self.vertical_layout.addWidget(self.num_io_slider)
        self.vertical_layout.addWidget(self.num_inputs_label)
        self.vertical_layout.addWidget(self.num_inputs_slider)
        self.vertical_layout.addWidget(self.num_unique_inputs_label)
        self.vertical_layout.addWidget(self.num_unique_inputs_slider)
        self.vertical_layout.addWidget(self.generate_button)

        self.setLayout(self.vertical_layout)

        # self.show()

    def set_up_section(self, text, single_step, minimum, maximum):
        label = QLabel(text + f": {minimum}", self)
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(minimum)
        slider.setSingleStep(single_step)

        slider.valueChanged.connect(
            lambda value: label.setText(
                " ".join(label.text().split()[:-1]) + " " + str(value)
            )
        )

        return label, slider

    def start_generating(self):  # TODO: error management
        num_comps_string = self.num_comps_line_edit.text()
        if not num_comps_string.isnumeric() or num_comps_string == "0":
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(
                self,
                "Error",
                "The number of compositions is expected to be a positive integer value.",
            )
            return

        if self.filename is None:
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(
                self, "Error", "No filename provided. Please select the desired path."
            )
            return

        args = Arguments(
            int(num_comps_string),
            self.filename,
            self.num_functions_slider.value(),
            self.num_io_slider.value(),
            self.num_inputs_slider.value(),
            self.num_unique_inputs_slider.value(),
            self.num_samples_per_comp_slider.value(),
            self.is_test_checkbox.isChecked(),
        )

        self.worker = DataWorker(args)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.bar_advanced.connect(self.update_bar)
        self.worker.finished.connect(self.finish_generating)
        self.thread.start()
        self.worker.start_signal.emit()
        self.generate_button.setEnabled(False)

    @pyqtSlot()
    def finish_generating(self):
        self.generate_button.setEnabled(True)
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.finished_generating.emit()

    def update_bar(self, value: int):
        self.bar_advanced.emit(value)

    def path_clicked(self):
        self.filename = QFileDialog.getSaveFileName(
            self, "Select path", "..", "DAT (*.dat)"
        )[0]
