import shlex
import signal
import subprocess
import webbrowser
from typing import Optional

from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QTextEdit


# noinspection PyUnresolvedReferences
class TensorBoardWorker(QObject):
    finished = pyqtSignal()
    start_signal = pyqtSignal()

    def __init__(self, cmd, server_running: bool):
        super().__init__()
        self.cmd = cmd
        self.server_running = server_running
        self.popen = None
        self.start_signal.connect(self.process, Qt.QueuedConnection)

    @pyqtSlot()
    def process(self):
        if not self.server_running:
            self.popen = subprocess.Popen(shlex.split(self.cmd))
        webbrowser.open("http://localhost:6006")
        self.finished.emit()


# noinspection PyUnresolvedReferences
class TrainDisplay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.thread = None
        self.worker = None
        self.server_running = False

        self.browser_button = QPushButton("Show details in browser", self)
        self.browser_button.clicked.connect(self.open_browser)
        self.vertical_layout = QVBoxLayout(self)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setTextInteractionFlags(Qt.NoTextInteraction)

        self.vertical_layout.addWidget(self.console)
        self.vertical_layout.addWidget(self.browser_button)

        self.setLayout(self.vertical_layout)

    def write_new_text(self, text):
        self.console.insertPlainText(text)

    def __del__(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        if self.worker is not None and self.worker.popen is not None:
            self.worker.popen.send_signal(signal.SIGTERM)

    @pyqtSlot()
    def open_browser(self):
        self.worker = TensorBoardWorker(
            "tensorboard --logdir lightning_logs", self.server_running
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.clean_up_worker)
        self.thread.start()
        self.worker.start_signal.emit()

        if not self.server_running:
            self.server_running = True

    @pyqtSlot()
    def clean_up_worker(self):
        self.worker = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None
