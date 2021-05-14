import os
import shlex
import signal
import subprocess
from typing import Optional

from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QCheckBox,
    QFileDialog,
    QPushButton,
)
from torch.cuda import device_count

from gui.base_option_list import BaseOptionList


# noinspection PyUnresolvedReferences
class TrainWorker(QObject):
    finished = pyqtSignal()
    start_signal = pyqtSignal()
    can_write = pyqtSignal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.popen = None
        self.start_signal.connect(self.process, Qt.QueuedConnection)
        self.interrupted = False

    @pyqtSlot()
    def process(self):
        self.popen = subprocess.Popen(
            shlex.split(self.cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        for stream in [self.popen.stderr, self.popen.stdout]:
            while line := stream.readline():
                self.can_write.emit(line.decode())

        self.finished.emit()


# noinspection PyUnresolvedReferences
class TrainOptionList(BaseOptionList):
    can_write = pyqtSignal(str)
    started_training = pyqtSignal()
    finished_training = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.load_path = None
        self.save_path = None
        self.thread = None
        self.worker = None

        self.vertical_layout = QVBoxLayout(self)
        self.select_dataset_button = QPushButton("Select dataset", self)
        self.select_dataset_button.clicked.connect(self.data_clicked)

        self.patience_label, self.patience_slider = self.set_up_section(
            "Patience of training", 1, 10
        )

        self.batch_size_checkbox = QCheckBox("Determine largest batch size", self)
        self.batch_size_checkbox.stateChanged.connect(self.checkbox_checked)

        self.batch_size_label, self.batch_size_slider = self.set_up_section(
            "Batch size", 5, 10, exp=True
        )

        num_gpus = device_count()
        self.num_gpus_label, self.num_gpus_slider = self.set_up_section(
            "Number of CUDA GPUs", 0, num_gpus
        )

        self.epochs_label, self.epochs_slider = self.set_up_section(
            "Maximum number of epochs", 1, 20, step_size=5
        )

        self.select_path_button = QPushButton("Select path", self)
        self.select_path_button.clicked.connect(self.path_clicked)

        self.train_button = QPushButton("Start training", self)
        self.train_button.clicked.connect(self.start_training)

        self.vertical_layout.addWidget(self.select_dataset_button)
        self.vertical_layout.addWidget(self.patience_label)
        self.vertical_layout.addWidget(self.patience_slider)
        self.vertical_layout.addWidget(self.batch_size_checkbox)
        self.vertical_layout.addWidget(self.batch_size_label)
        self.vertical_layout.addWidget(self.batch_size_slider)
        self.vertical_layout.addWidget(self.num_gpus_label)
        self.vertical_layout.addWidget(self.num_gpus_slider)
        self.vertical_layout.addWidget(self.epochs_label)
        self.vertical_layout.addWidget(self.epochs_slider)
        self.vertical_layout.addWidget(self.select_path_button)
        self.vertical_layout.addWidget(self.train_button)

        self.train_button.setFixedHeight(50)
        self.train_button.setFont(QFont("Roboto", 15))
        self.setFixedWidth(300)

    @pyqtSlot()
    def data_clicked(self):
        self.load_path = QFileDialog.getOpenFileName(
            self, "Select dataset", ".", "DAT (*.dat)"
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

    @pyqtSlot()
    def start_training(self):
        if self.train_button.text() == "Start training":
            if self.load_path is None:
                self.warn("No dataset provided. Please select the desired path.")
                return

            if self.save_path is None:
                self.warn("No save path provided.")
                return

            self.train_button.setText("Stop")
            self.train_button.setStyleSheet(
                """
                :!hover{background-color: #c00000;}
                :hover{background-color: #ff0000;}
                """
            )

            cmd = (
                f"python train.py {'--auto_scale_batch_size power' if self.batch_size_checkbox.isChecked() else ''} "
                f"--gpus {self.num_gpus_slider.value()} --dataset {self.load_path} --save_path {self.save_path} "
                f"--max_epochs {5*self.epochs_slider.value()} "
                f"--batch_size {2**self.batch_size_slider.value()} --patience {self.patience_slider.value()}"
            )

            self.start_worker(cmd)

        else:
            if self.worker.popen is not None:
                self.worker.popen.send_signal(signal.SIGTERM)

    @pyqtSlot(str)
    def pass_string(self, value: str):
        self.can_write.emit(value)

    @pyqtSlot()
    def reactivate_button(self):
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.train_button.setText("Start training")
        self.train_button.setStyleSheet("")
        self.finished_training.emit()

    @pyqtSlot(int)
    def checkbox_checked(self, _: int):
        if self.batch_size_slider.isEnabled():
            self.batch_size_slider.setValue(self.batch_size_slider.minimum())
            self.batch_size_slider.setEnabled(False)
        else:
            self.batch_size_slider.setEnabled(True)

    def start_worker(self, cmd):
        self.worker = TrainWorker(cmd)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.can_write.connect(self.pass_string)
        self.worker.finished.connect(self.reactivate_button)
        self.thread.start()
        self.worker.start_signal.emit()
        self.started_training.emit()

    def __del__(self):
        if self.worker is not None and self.worker.sub_pid is not None:
            os.kill(self.worker.sub_pid, signal.CTRL_C_EVENT)
        self.worker = None
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
