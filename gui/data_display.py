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
        super().__init__(fig)
        fig.set_facecolor("#303030")
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor("#303030")
        matplotlib.rcParams["text.color"] = "white"
        matplotlib.rcParams["xtick.color"] = "white"
        matplotlib.rcParams["ytick.color"] = "white"
        matplotlib.rcParams["axes.labelcolor"] = "white"


class DataDisplay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.mpl_canvas = MplCanvas()
        self.populate_diagram()

        self.vertical_layout = QVBoxLayout(self)
        self.vertical_layout.addWidget(self.mpl_canvas)

        self.setLayout(self.vertical_layout)

    def populate_diagram(self, statistics: Optional[dict] = None):
        if statistics is None:
            statistics = MOCK_DICT
        self.mpl_canvas.axes.clear()
        self.mpl_canvas.axes.set_title("Distribution of function types")
        self.mpl_canvas.axes.bar(*zip(*statistics.items()), color="green")

        for tick in self.mpl_canvas.axes.get_xticklabels():
            tick.set_rotation(90)
        self.mpl_canvas.draw()
