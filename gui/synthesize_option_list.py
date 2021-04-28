import json
from ast import literal_eval
from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
)

from evaluator import SynthesizeWorker


# noinspection PyUnresolvedReferences
class SynthesizeOptionList(QWidget):
    bar_advanced = pyqtSignal(int)
    finished_synthesizing = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.load_path = None
        self.save_path = None
        self.model_filename = None
        self.thread = None
        self.worker = None

        self.vertical_layout = QVBoxLayout(self)

        self.select_model_button = QPushButton("Select model", self)
        self.select_model_button.clicked.connect(self.model_clicked)

        self.select_dataset_button = QPushButton("Select test dataset", self)
        self.select_dataset_button.clicked.connect(self.data_clicked)

        self.select_path_button = QPushButton("Select save path", self)
        self.select_path_button.clicked.connect(self.path_clicked)

        self.input_label = QLabel("Input(s)")
        self.input_label.setFixedHeight(30)
        self.input_label.setFont(QFont("Roboto", 15))

        self.input_line_edit = QLineEdit()
        self.input_line_edit.textChanged.connect(self.text_changed)

        self.output_label = QLabel("Output(s)")
        self.output_label.setFixedHeight(30)
        self.output_label.setFont(QFont("Roboto", 15))
        self.output_line_edit = QLineEdit()
        self.output_line_edit.textChanged.connect(self.text_changed)

        self.synthesize_button = QPushButton("Start synthesizing", self)
        self.synthesize_button.clicked.connect(self.start_synthesizing)
        self.synthesize_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.synthesize_button.setFixedHeight(50)
        self.synthesize_button.setFont(QFont("Roboto", 15))

        self.vertical_layout.addWidget(self.select_model_button)
        self.vertical_layout.addWidget(self.select_dataset_button)
        self.vertical_layout.addWidget(self.select_path_button)
        self.vertical_layout.addWidget(self.input_label)
        self.vertical_layout.addWidget(self.input_line_edit)
        self.vertical_layout.addWidget(self.output_label)
        self.vertical_layout.addWidget(self.output_line_edit)
        self.vertical_layout.addWidget(self.synthesize_button)

        self.setLayout(self.vertical_layout)

        self.setFixedWidth(250)

        # self.show()

    def text_changed(self):
        if self.input_line_edit.text() == self.output_line_edit.text() == "":
            self.select_dataset_button.setEnabled(True)
        else:
            self.select_dataset_button.setEnabled(False)
            self.load_path = None

    def start_synthesizing(self):
        if self.input_line_edit.text() == "" and self.output_line_edit.text() != "":
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(self, "Error", "Please provide the inputs, too.")
            return

        if self.output_line_edit.text() == "" and self.input_line_edit.text() != "":
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(self, "Error", "Please provide the outputs, too.")
            return

        if (
            self.input_line_edit.text() == self.output_line_edit.text() == ""
            and self.load_path is None
        ):
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(
                self, "Error", "Please provide the dataset or exact problem."
            )
            return

        if self.model_filename is None:
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(self, "Error", "Please provide a model checkpoint.")
            return

        if self.save_path is None:
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(
                self,
                "Error",
                "Please provide the save path of the synthesized compositions.",
            )
            return

        if self.input_line_edit.text() == self.output_line_edit.text() == "":
            load_filename = self.load_path
        else:
            try:
                input_state_tuple = literal_eval(self.input_line_edit.text())

                if not (
                    isinstance(input_state_tuple, tuple)
                    and all(isinstance(elem, list) for elem in input_state_tuple)
                    and (
                        all(
                            isinstance(num, int)
                            for elem in input_state_tuple
                            for num in elem
                        )
                        or (
                            all(
                                isinstance(subelem, list)
                                for elem in input_state_tuple
                                for subelem in elem
                            )
                            and all(
                                isinstance(subsubelem, int)
                                for elem in input_state_tuple
                                for subelem in elem
                                for subsubelem in subelem
                            )
                        )
                    )
                ):
                    raise ValueError
            except (SyntaxError, ValueError):
                warning_screen = QMessageBox()
                warning_screen.setFixedSize(500, 200)
                warning_screen.critical(
                    self, "Error", "Invalid input state tuple provided."
                )
                return

            try:
                output_list = literal_eval(self.output_line_edit.text())

                if not (
                    all(isinstance(num, int) for num in output_list)
                    or (
                        all(isinstance(sublist, list) for sublist in output_list)
                        and all(
                            isinstance(num, int)
                            for sublist in output_list
                            for num in sublist
                        )
                    )
                ):
                    raise ValueError
            except (SyntaxError, ValueError):
                warning_screen = QMessageBox()
                warning_screen.setFixedSize(500, 200)
                warning_screen.critical(
                    self, "Error", "Invalid output state tuple provided."
                )
                return

            sample_dict = {"input": input_state_tuple, "output": [output_list]}
            with open("sample000.dat", "w") as f:
                json.dump(sample_dict, f)
            load_filename = "sample000.dat"

        self.worker = SynthesizeWorker(
            load_filename, self.save_path, self.model_filename
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.bar_advanced.connect(self.update_bar)
        self.worker.finished.connect(self.finish_synthesizing)
        self.thread.start()
        self.worker.start_signal.emit()
        self.synthesize_button.setEnabled(False)

    @pyqtSlot()
    def finish_synthesizing(self):
        self.synthesize_button.setEnabled(True)
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.finished_synthesizing.emit()

    def update_bar(self, value: int):
        self.bar_advanced.emit(value)

    def model_clicked(self):
        self.model_filename = QFileDialog.getOpenFileName(
            self, "Select model", "..", "CKPT (*.ckpt)"
        )[0]

    def data_clicked(self):
        self.load_path = QFileDialog.getOpenFileName(
            self, "Select test dataset", "..", "DAT (*.dat)"
        )[0]

    def path_clicked(self):
        self.save_path = QFileDialog.getExistingDirectory(
            self, "Select path", "..", QFileDialog.ShowDirsOnly
        )
