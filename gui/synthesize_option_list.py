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
    QSizePolicy,
)

from evaluator import SynthesizeWorker
from gui.base_option_list import BaseOptionList


# noinspection PyUnresolvedReferences
class SynthesizeOptionList(BaseOptionList):
    bar_advanced = pyqtSignal(int)
    started_synthesizing = pyqtSignal()
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

    @pyqtSlot()
    def text_changed(self):
        if self.input_line_edit.text() == self.output_line_edit.text() == "":
            self.select_dataset_button.setEnabled(True)
        else:
            self.select_dataset_button.setEnabled(False)
            self.load_path = None

    @pyqtSlot()
    def start_synthesizing(self):
        if self.synthesize_button.text() == "Start synthesizing":
            if not self.inspect_synthesizing_args():
                return

            if self.input_line_edit.text() == self.output_line_edit.text() == "":
                load_filename = self.load_path
            else:
                ret = self.inspect_input_state_tuple()
                if ret is None:
                    return
                input_state_tuple, input_io = ret

                output_list = self.inspect_output_list(input_io)
                if output_list is None:
                    return

                sample_dict = {"input": input_state_tuple, "output": [output_list]}
                with open(".sample.dat", "w") as f:
                    json.dump(sample_dict, f)
                load_filename = ".sample.dat"

            self.start_worker(load_filename)

            self.synthesize_button.setText("Stop")
            self.synthesize_button.setStyleSheet(
                """
                :!hover{background-color: #c00000;}
                :hover{background-color: #ff0000;}
                """
            )
        else:
            self.worker.shutdown = True

    def inspect_synthesizing_args(self) -> bool:
        if self.input_line_edit.text() == "" and self.output_line_edit.text() != "":
            self.warn("Please provide the inputs, too.")
            return False

        if self.output_line_edit.text() == "" and self.input_line_edit.text() != "":
            self.warn("Please provide the outputs, too.")
            return False

        if (
            self.input_line_edit.text() == self.output_line_edit.text() == ""
            and self.load_path is None
        ):
            self.warn("Please provide a dataset or an exact problem.")
            return False

        if self.model_filename is None:
            self.warn("Please provide a model checkpoint.")
            return False

        if self.save_path is None:
            self.warn("Please provide the save path of the synthesized compositions.")
            return False

        return True

    def inspect_input_state_tuple(self) -> Optional[tuple]:
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
            self.warn("Invalid input state tuple provided.")
            return None

        if (
            not input_state_tuple
            or any(not elem for elem in input_state_tuple)
            or any(subelem == [] for elem in input_state_tuple for subelem in elem)
        ):
            self.warn("Empty inputs are not allowed.")
            return None

        if isinstance(input_state_tuple[0][0], list):
            input_io = len(input_state_tuple[0])
            if input_io == 1:
                self.warn("Remove unnecessary parentheses from input.")
                return None

            if not all(len(elem) == input_io for elem in input_state_tuple):
                self.warn("Mismatching number of I/O examples provided.")
                return None
        else:
            input_io = 1

        return input_state_tuple, input_io

    def inspect_output_list(self, input_io) -> list:
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
            self.warn("Invalid output state tuple provided.")
            return None

        if all(elem == [] for elem in output_list):
            self.warn("Cases where all of the outputs are empty are not allowed.")
            return None

        if isinstance(output_list[0], list):
            output_io = len(output_list)
            if output_io == 1:
                self.warn("Remove unnecessary parentheses from output.")
                return None

        else:
            output_io = 1

        if input_io != output_io:
            self.warn("Mismatching number of I/O examples provided.")
            return None

        return output_list

    @pyqtSlot()
    def finish_synthesizing(self):
        self.synthesize_button.setText("Start synthesizing")
        self.synthesize_button.setStyleSheet("")
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.finished_synthesizing.emit()

    @pyqtSlot(int)
    def update_bar(self, value: int):
        self.bar_advanced.emit(value)

    @pyqtSlot()
    def model_clicked(self):
        self.model_filename = QFileDialog.getOpenFileName(
            self, "Select model", ".", "CKPT (*.ckpt)"
        )[0]
        if not self.model_filename:
            self.model_filename = None

    @pyqtSlot()
    def data_clicked(self):
        self.load_path = QFileDialog.getOpenFileName(
            self, "Select test dataset", ".", "DAT (*.dat)"
        )[0]
        if not self.load_path:
            self.load_path = None

    @pyqtSlot()
    def path_clicked(self):
        self.save_path = QFileDialog.getExistingDirectory(
            self, "Select path", ".", QFileDialog.ShowDirsOnly
        )
        if not self.save_path:
            self.save_path = None

    def start_worker(self, load_filename):
        self.worker = SynthesizeWorker(
            load_filename, self.save_path, self.model_filename
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.bar_advanced.connect(self.update_bar)
        self.worker.finished.connect(self.finish_synthesizing)
        self.thread.start()
        self.worker.start_signal.emit()
        self.started_synthesizing.emit()
