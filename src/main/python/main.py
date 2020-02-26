from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets as widgets

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

    
if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = App()
    window.show()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
