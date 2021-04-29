import json
import math

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

import flexcoder
from src.utils import visualize


class SynthesizeWorker(QObject):
    bar_advanced = pyqtSignal(int)
    start_signal = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, load_filename, save_filename, model_filename):
        super().__init__()
        self.load_filename = load_filename
        self.save_filename = save_filename
        self.model_filename = model_filename
        self.start_signal.connect(self.process, Qt.QueuedConnection)
        self.shutdown = False

    @pyqtSlot()
    def process(self):
        flexcoder.model = flexcoder.ModelFacade(self.model_filename)
        with open(self.load_filename) as f:
            lines = f.readlines()
            len_lines = len(lines)

            counter = 0
            png_counter = 1
            file_image = []

            for i, line in enumerate(lines):
                out = json.loads(line)
                inp = tuple([[x]] for x in out["input"])
                res = flexcoder.beam_search(inp, 100, 8, out["output"], self)

                if self.shutdown:
                    with open("images.json", "w") as json_file:
                        json.dump(file_image, json_file)
                    self.finished.emit()
                    return

                if res["program"] is not None:
                    counter += 1
                    filename = f"{self.save_filename}/result{png_counter}.png"
                    visualize(res["composition"], filename)
                    file_image.append(
                        {
                            "input": out["input"],
                            "output": out["output"],
                            "path": filename,
                        }
                    )
                else:
                    file_image.append(
                        {"input": out["input"], "output": out["output"], "path": None}
                    )
                png_counter += 1
                self.bar_advanced.emit(math.ceil(100 * i / len_lines))

            with open("images.json", "w") as f:
                json.dump(file_image, f)

            self.finished.emit()
