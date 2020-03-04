from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets as widgets

import pyqtgraph as pg

import sys


class App(widgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

    def init_ui(self):
        self.setWindowTitle(self.title)
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        self.display_viewer()

    def display_viewer(self):
        self.main_view = SpectrogramVisualizer(self)
        self.setCentralWidget(self.main_view)
        self.show()


class SpectrogramVisualizer(widgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.groupbox1 = widgets.QGroupBox("Scatter 2D")

        self.plot = pg.PlotWidget()

        pen = pg.mkPen((200, 200, 250, 127))

        self.scatter = pg.ScatterPlotItem(pen=pen, symbol="o", size=1)

        self.plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

        self.scatter.hoverEvent = None
        self.scatter.mouseClickEvent = None

        self.plot.addItem(self.scatter)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addStretch(1)
        self.groupbox1.setLayout(layout)

    
if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = App()
    window.show()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
