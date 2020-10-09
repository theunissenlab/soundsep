"""File for audio viewing and interaction components
"""
from functools import partial

import pyqtgraph as pg

from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5 import QtCore
from PyQt5 import QtGui as gui


class SpectrogramViewBox(pg.ViewBox):
    """SpectrogramViewBox manages user interactions with the spectrogram box

    Emits the following events
    ==========================
    dragInProgress(QPointF, QPointF)
        Emitted as mouse is held down and dragged over the spectrogram.
        Sends the start position and end position as QtCore.QPointF objects
        indicating the start and end positions of the drag line in plot
        coordinates (i.e. ith sample, jth frequency bin)

    dragComplete(QPointF, QPointF)
        Emitted as mouse released after being dragged over spectrogram.
        Sends the start position and end position as QtCore.QPointF objects
        indicating the start and end positions of the drag line in plot
        coordinates (i.e. ith sample, jth frequency bin)

    clicked(QPointF, event)
        Emitted when mouse is clicked and released (not held and dragged).
        Sends the click position in plot coordinates (ith sample, jth freq bin),
        as well as the original click event object.

    menuSelection(object)
        Emitted when a context menu item (right click menu item) is selected.
        Sends the string identifier for the menu item. Currently these are
        'search', 'delete', 'merge', 'desperate', 'zoom_in', and 'zoom_out',
        though these may change.

    zoomEvent(int, object)
        Emitted when the scroll wheel is rolled. Emits the direction of scroll
        as an integer (+1 or -1) and a tuple of the x,y position of the cursor
        during the event in fraction of the window (x and y range from 0-1)
    """
    dragComplete = pyqtSignal(QtCore.QPointF, QtCore.QPointF)
    dragInProgress = pyqtSignal(QtCore.QPointF, QtCore.QPointF)
    clicked = pyqtSignal(QtCore.QPointF)
    menuSelection = pyqtSignal(object)
    zoomEvent = pyqtSignal(int, object)

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
            self.search = gui.QAction("Search", self.menu)
            self.delete = gui.QAction("Delete Selected Interval (Del)", self.menu)
            self.merge = gui.QAction("Merge", self.menu)
            self.desperate = gui.QAction("I'm desperate", self.menu)

            self.zoom_to_selection = gui.QAction("Zoom to Selection", self.menu)
            self.zoom_out_fully = gui.QAction("Zoom Out", self.menu)

            self.search.triggered.connect(partial(self._emit_menu_selection, "search"))
            self.delete.triggered.connect(partial(self._emit_menu_selection, "delete"))
            self.merge.triggered.connect(partial(self._emit_menu_selection, "merge"))
            self.desperate.triggered.connect(partial(self._emit_menu_selection, "desperate"))

            self.zoom_to_selection.triggered.connect(partial(self._emit_menu_selection, "zoom_in"))
            self.zoom_out_fully.triggered.connect(partial(self._emit_menu_selection, "zoom_out"))

            self.menu.addAction(self.search)
            self.menu.addAction(self.merge)
            self.menu.addAction(self.delete)
            self.menu.addAction(self.desperate)
            self.menu.addSeparator()
            self.menu.addAction(self.zoom_to_selection)
            self.menu.addAction(self.zoom_out_fully)

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

    def wheelEvent(self, event):
        """Emits the direction of scroll and the location in fractional position"""
        pos = self.mapSceneToView(event.scenePos())
        xmax = self.viewRange()[0][1]
        ymax = self.viewRange()[1][1]
        xy = (pos.x() / xmax, pos.y() / ymax)
        if event.delta() > 0:
            self.zoomEvent.emit(1, xy)
        elif event.delta() < 0:
            self.zoomEvent.emit(-1, xy)
