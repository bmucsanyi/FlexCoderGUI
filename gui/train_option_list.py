import shlex
import subprocess
from typing import Optional

from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QSlider,
    QFileDialog,
    QPushButton,
    QMessageBox,
)
from torch.cuda import device_count


class TrainWorker(QObject):
    finished = pyqtSignal()
    start_signal = pyqtSignal()
    can_write = pyqtSignal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.sub_pid = None
        self.start_signal.connect(self.process, Qt.QueuedConnection)

    @pyqtSlot()
    def process(self):
        process = subprocess.Popen(
            shlex.split(self.cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        for stream in [process.stderr, process.stdout]:
            while line := stream.readline():
                self.can_write.emit(line.decode())

        self.finished.emit()


# noinspection PyUnresolvedReferences
class TrainOptionList(QWidget):
    can_write = pyqtSignal(str)

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

        # self.show()

    def set_up_section(self, text, minimum, maximum, step_size=1, exp=False):
        if exp:
            label = QLabel(text + f": {2**minimum}", self)
        else:
            label = QLabel(text + f": {minimum * step_size}", self)
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(minimum)

        if exp:
            slider.valueChanged.connect(
                lambda value: label.setText(
                    " ".join(label.text().split()[:-1]) + " " + str(2 ** value)
                )
            )
        else:
            slider.valueChanged.connect(
                lambda value: label.setText(
                    " ".join(label.text().split()[:-1]) + " " + str(value * step_size)
                )
            )

        return label, slider

    def data_clicked(self):
        self.load_path = QFileDialog.getOpenFileName(
            self, "Select dataset", "..", "DAT (*.dat)"
        )[0]

    def path_clicked(self):
        self.save_path = QFileDialog.getExistingDirectory(
            self, "Select path", "..", QFileDialog.ShowDirsOnly
        )

    def start_training(self):
        if self.load_path is None:
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(
                self, "Error", "No dataset provided. Please select the desired path."
            )
            return

        if self.save_path is None:
            warning_screen = QMessageBox()
            warning_screen.setFixedSize(500, 200)
            warning_screen.critical(self, "Error", "No save path provided.")
            return

        self.train_button.setEnabled(False)
        cmd = (
            f"python train.py {'--auto_scale_batch_size power' if self.batch_size_checkbox.isChecked() else ''} "
            f"--gpus {self.num_gpus_slider.value()} --dataset {self.load_path} --save_path {self.save_path} "
            f"--max_epochs {5*self.epochs_slider.value()} "
            f"--batch_size {2**self.batch_size_slider.value()} --patience {self.patience_slider.value()}"
        )

        self.worker = TrainWorker(cmd)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.can_write.connect(self.pass_string)
        self.worker.finished.connect(self.reactivate_button)
        self.thread.start()
        self.worker.start_signal.emit()

    def pass_string(self, value: str):
        self.can_write.emit(value)

    def reactivate_button(self):
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.train_button.setEnabled(True)

    def checkbox_checked(self, _: int):
        if self.batch_size_slider.isEnabled():
            self.batch_size_slider.setValue(self.batch_size_slider.minimum())
            self.batch_size_slider.setEnabled(False)
        else:
            self.batch_size_slider.setEnabled(True)
