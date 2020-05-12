"""File for audio viewing and interaction components
"""
from functools import partial

import pyqtgraph as pg

from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5 import QtCore
from PyQt5 import QtGui as gui


class SpectrogramViewBox(pg.ViewBox):
    """docstring for SpectrogramViewBox."""
    dragComplete = pyqtSignal(QtCore.QPointF, QtCore.QPointF)
    dragInProgress = pyqtSignal(QtCore.QPointF, QtCore.QPointF)
    clicked = pyqtSignal(QtCore.QPointF)
    menuSelection = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.menu = None
        self.menu = self.getMenu() # Create the menu

    def raiseContextMenu(self, ev):
        # if not self.menuEnabled():
        #     return
        menu = self.getMenu()
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))

    def getMenu(self):
        if self.menu is None:
            self.menu = gui.QMenu()
            self.search_more = gui.QAction("Search More (W)", self.menu)
            self.search_less = gui.QAction("Search Less (S)", self.menu)
            self.menu.addSeparator()
            self.delete = gui.QAction("Delete Selected Interval (Del)", self.menu)
            self.menu.addSeparator()
            self.clear_all_on_channel = gui.QAction("Clear Entire Channel", self.menu)

            self.search_more.triggered.connect(partial(self._emit_menu_selection, "search+"))
            self.search_less.triggered.connect(partial(self._emit_menu_selection, "search-"))
            self.delete.triggered.connect(partial(self._emit_menu_selection, "delete"))
            self.clear_all_on_channel.triggered.connect(partial(self._emit_menu_selection, "clear_all"))

            self.menu.addAction(self.search_more)
            self.menu.addAction(self.search_less)
            self.menu.addAction(self.delete)
            self.menu.addAction(self.clear_all_on_channel)

        return self.menu

    def _emit_menu_selection(self, item):
        self.menuSelection.emit(item)

    def mouseDragEvent(self, event):
        event.accept()
        start_pos = self.mapSceneToView(event.buttonDownScenePos())
        end_pos = self.mapSceneToView(event.scenePos())
        if event.isFinish():
            self.dragComplete.emit(start_pos, end_pos)
        else:
            self.dragInProgress.emit(start_pos, end_pos)

    def mouseClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
            self.clicked.emit(self.mapSceneToView(event.scenePos()))
        else:
            super().mouseClickEvent(event)
