from typing import Optional

import matplotlib
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from src.grammar import DEFINITIONS, ABBREVATION_DICT

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


MOCK_DICT = {ABBREVATION_DICT[func_name]: 0 for func_name in DEFINITIONS[:-1]}


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class DataDisplay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.mpl_canvas = MplCanvas()
        self.populate_diagram(MOCK_DICT)

        self.vertical_layout = QVBoxLayout(self)
        self.vertical_layout.addWidget(self.mpl_canvas)

        self.setLayout(self.vertical_layout)
        # self.show()

    def populate_diagram(self, statistics: dict):
        self.mpl_canvas.axes.clear()
        self.mpl_canvas.axes.set_title("Distribution of function types")
        self.mpl_canvas.axes.bar(*zip(*statistics.items()))

        for tick in self.mpl_canvas.axes.get_xticklabels():
            tick.set_rotation(90)
        self.mpl_canvas.draw()


# app = QApplication(sys.argv)
# w = TrainDisplay()
# app.exec_()
