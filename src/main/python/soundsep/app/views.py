import pyqtgraph as pg
import numpy as np

from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets

from soundsig.sound import spectrogram

from app.components import SpectrogramViewBox
from app.state import AppState, ViewState
from app.settings import fonts, read_default
from app.style import qss


class MainView(widgets.QWidget):
    """Container for the overall layout of the app

    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.events = self.parent().events
        self.state = AppState()

        self.init_ui()
        self.init_nested_views()

        self.events.dataLoaded.connect(self.on_data_load)
        self.events.createSource.connect(self.on_create_source)

    def init_ui(self):
        self.topBar = widgets.QHBoxLayout()
        self.topBarLabel = widgets.QLabel("")
        self.topBarLabel.setDisabled(True)
        self.topBar.addWidget(self.topBarLabel)
        self.topBar.addStretch(1)

        self.topLeftBox = widgets.QGroupBox("Sources")
        self.topRightBox = widgets.QGroupBox("Sound Viewer")
        self.bottomBox = widgets.QGroupBox("Control Panel")

        self.topSplitter = widgets.QSplitter(Qt.Horizontal)
        self.topSplitter.addWidget(self.topLeftBox)
        self.topSplitter.addWidget(self.topRightBox)
        self.topSplitter.setStretchFactor(0, 0)
        self.topSplitter.setStretchFactor(1, 1)

        self.mainLayout = widgets.QGridLayout()
        self.mainLayout.addLayout(self.topBar, 0, 0)
        self.mainLayout.addWidget(self.topSplitter, 1, 0, 1, 6)
        self.mainLayout.addWidget(self.bottomBox, 2, 0, 1, 6)

        self.mainLayout.setRowStretch(0, 0)
        self.mainLayout.setRowStretch(1, 4)
        self.mainLayout.setRowStretch(2, 0)

        for col in range(6):
            self.mainLayout.setColumnStretch(col, 1)

        self.setLayout(self.mainLayout)

        self.topLeftBox.setDisabled(True)
        self.topRightBox.setDisabled(True)

    def init_nested_views(self):
        self.source_manager_view = SourceManagerView(
            None,
            events=self.events,
        )
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.source_manager_view)
        self.topLeftBox.setLayout(layout)

        self.audio_view = AudioView(
            None,
            events=self.events,
        )
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.audio_view)
        self.topRightBox.setLayout(layout)

    def on_create_source(self, source_dict):
        sources = self.state.get("sources")
        if sources is None:
            self.state.set("sources", [])
            sources = self.state.get("sources")
        # check if name is unique
        if source_dict["name"] in [s["name"] for s in sources]:
            widgets.QMessageBox.warning(
                self,
                "Error",
                "Source name already in use.",
            )
            return

        sources.append({
            "name": source_dict["name"],
            "channel": source_dict["channel"],
            "visible": True
        })
        self.events.sourcesChanged.emit()

    def on_data_load(self):
        self.topLeftBox.setDisabled(False)
        self.topRightBox.setDisabled(False)
        self.topBarLabel.setText(self.state.get("sound_file"))


class SourceManagerView(widgets.QScrollArea):

    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.state = AppState()
        self.events = events

        self.init_ui()

        self.events.sourcesChanged.connect(self.on_sources_changed)
        self.events.dataLoaded.connect(self.on_sources_changed)

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()

        # self.headingLayout = widgets.QHBoxLayout()
        # source_label = widgets.QLabel("Source Name")
        # source_label.setFont(fonts.subheading)
        # self.headingLayout.addWidget(source_label)
        # channel_label = widgets.QLabel("Channel")
        # channel_label.setFont(fonts.subheading)
        # self.headingLayout.addWidget(channel_label)

        self.currentSourcesLayout = widgets.QVBoxLayout()
        self.addSourceButton = widgets.QToolButton()
        addIcon = gui.QIcon("images/plus_icon.svg")
        self.addSourceButton = widgets.QToolButton(self)
        self.addSourceButton.setIcon(addIcon)
        self.addSourceButton.setToolTip("Create a new source")
        self.addSourceButton.setFixedWidth(50)
        self.addSourceButton.clicked.connect(self.show_add_source_window)

        # self.mainLayout.addLayout(self.headingLayout)
        self.mainLayout.addLayout(self.currentSourcesLayout)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.addSourceButton)
        self.setLayout(self.mainLayout)

    def on_sources_changed(self):
        for i in reversed(range(self.currentSourcesLayout.count())):
            item = self.currentSourcesLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        for i, source in enumerate(self.state.get("sources")):
            new_widget = SourceEditorView(
                self,
                source_idx=i,
                events=self.events)
            self.currentSourcesLayout.addWidget(new_widget)

    def show_add_source_window(self):
        self.dialog = widgets.QDialog(self)

        layout = widgets.QGridLayout()
        nameLabel = widgets.QLabel(self, text="Name")
        self.nameInput = widgets.QLineEdit(self, text="")
        layout.addWidget(nameLabel, 0, 0)
        layout.addWidget(self.nameInput, 0, 1)

        channelLabel = widgets.QLabel(self, text="Channel")
        self.channelInput = widgets.QComboBox(self)
        for ch in range(self.state.get("sound_object").n_channels):
            self.channelInput.addItem(str(ch), ch)
        layout.addWidget(channelLabel, 1, 0)
        layout.addWidget(self.channelInput, 1, 1)

        submitButton = widgets.QPushButton("Submit")
        layout.addWidget(submitButton, 2, 1)
        submitButton.clicked.connect(self.submit)

        self.dialog.setLayout(layout)
        self.dialog.setWindowTitle("Create Source")
        self.dialog.setMinimumWidth(500)
        self.dialog.setMaximumWidth(self.dialog.width())
        self.dialog.setMaximumHeight(self.dialog.height())
        self.dialog.show()

    def submit(self):
        name = self.nameInput.text()
        channel = self.channelInput.currentIndex()

        self.events.createSource.emit({
            "name": name,
            "channel": channel
        })
        self.dialog.close()


class SourceEditorView(widgets.QWidget):
    """Display of source information and options to edit/delete/hide/show
    """
    editModeChanged = pyqtSignal(bool)

    def __init__(self, parent=None, source_idx=None, events=None):
        super().__init__(parent)
        self.events = events
        self.state = AppState()
        self.source_idx = source_idx
        self.source = self.state.get("sources")[self.source_idx]
        self.setStyleSheet(qss)
        self.init_ui()

    def init_ui(self):
        self.mainLayout = widgets.QHBoxLayout(self)

        self.nameInput = widgets.QLineEdit(self, text=self.source["name"])
        self.nameInput.setStyleSheet("QLineEdit { qproperty-cursorPosition: 0; }");
        self.channelInput = widgets.QComboBox(self)
        for ch in range(self.state.get("sound_object").n_channels):
            self.channelInput.addItem(str(ch), ch)
        self.channelInput.setCurrentIndex(self.source["channel"])
        self.nameInput.setDisabled(True)
        self.channelInput.setDisabled(True)

        editIcon = gui.QIcon("images/edit_icon.svg")
        self.editButton = widgets.QToolButton(self)
        self.editButton.setIcon(editIcon)
        self.editButton.setCheckable(True)
        deleteIcon = gui.QIcon("images/delete_icon.svg")
        self.deleteButton = widgets.QToolButton(self)
        self.deleteButton.setIcon(deleteIcon)

        self.nameInput.returnPressed.connect(self.editButton.click)
        self.editButton.clicked.connect(self.toggle_edit_mode)
        self.deleteButton.clicked.connect(self.delete_source)

        self.mainLayout.addWidget(self.nameInput)
        self.mainLayout.addWidget(self.channelInput)
        self.mainLayout.addWidget(self.editButton)
        self.mainLayout.addWidget(self.deleteButton)

        self.setLayout(self.mainLayout)

    def toggle_edit_mode(self):
        if self.editButton.isChecked():
            self.nameInput.setDisabled(False)
            self.channelInput.setDisabled(False)
        else:
            self.nameInput.setDisabled(True)
            self.channelInput.setDisabled(True)
            self.repaint()
            new_name = self.nameInput.text()
            new_channel = self.channelInput.currentIndex()
            if self.source["name"] == new_name and self.source["channel"] == new_channel:
                return
            else:
                self.source["name"] = new_name
                self.source["channel"] = new_channel
                self.events.sourcesChanged.emit()

    def delete_source(self):
        msg = ("Are you sure you want to delete source \"{}\"?\n"
            "This cannot be undone. All vocal periods labeled for this\n"
            "source will be deleted. Permanently.").format(self.source["name"])
        reply = widgets.QMessageBox.question(
                self,
                'Delete Confirmation',
                msg,
                widgets.QMessageBox.Yes,
                widgets.QMessageBox.No)
        if reply == widgets.QMessageBox.No:
            return

        del self.state.get("sources")[self.source_idx]
        self.events.sourcesChanged.emit()


class AudioView(widgets.QWidget):

    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.events = events
        self.state = AppState()
        self.view_state = ViewState()
        self.settings = QSettings("Theuniseen Lab", "Sound Separation")

        self.init_ui()

        self.state.set("lowres_preview", False)
        self.source_view_registry = {}

        self.events.sourcesChanged.connect(self.on_sources_changed)
        self.events.rangeChanged.connect(self.on_range_changed)
        self.events.dataLoaded.connect(self.on_data_loaded)

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()

        self.topBarLayout = widgets.QHBoxLayout()
        self.currentSourcesLayout = widgets.QVBoxLayout()

        self.scrollbar = widgets.QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(100)

        self.scrollbar.sliderPressed.connect(self.on_slider_press)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_change)
        self.scrollbar.sliderReleased.connect(self.on_slider_release)

        self.mainLayout.addLayout(self.topBarLayout)
        self.mainLayout.addLayout(self.currentSourcesLayout)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.scrollbar)

        self.setLayout(self.mainLayout)

    def _update_scroll_bar(self):
        win_size = read_default(self.settings, "WINDOW_SIZE")
        t_last = self.state.get("sound_object").t_max - win_size

        page_step = read_default(self.settings, "PAGE_STEP")
        steps = (int(np.ceil(t_last / win_size))) * page_step
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(steps)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setPageStep(page_step)

    def on_slider_press(self):
        self.state.set("lowres_preview", True)

    def on_slider_release(self):
        self.state.set("lowres_preview", False)
        self.on_scrollbar_value_change(self.scrollbar.value())

    def on_scrollbar_value_change(self, new_value):
        win_size = read_default(self.settings, "WINDOW_SIZE")
        page_step = read_default(self.settings, "PAGE_STEP")
        t_step = win_size / page_step

        t1 = t_step * new_value
        t2 = t1 + win_size
        t2 = min(t2, self.state.get("sound_object").t_max)

        self.view_state.set("current_range", (t1, t2))
        self.events.rangeChanged.emit()

    def on_data_loaded(self):
        self._update_scroll_bar()
        self.on_scrollbar_value_change(0)
        self.on_sources_changed()

    def on_sources_changed(self):
        for i in reversed(range(self.currentSourcesLayout.count())):
            item = self.currentSourcesLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        self.source_view_registry = []
        for i, source in enumerate(self.state.get("sources")):
            source_view = SourceView(source_idx=i, events=self.events)
            source_label = widgets.QLabel(source["name"])
            source_label.setFont(fonts.subheading)
            self.currentSourcesLayout.addWidget(source_label)
            self.currentSourcesLayout.addWidget(source_view)
            self.source_view_registry.append(source_view)

        self.on_redraw()

    def on_range_changed(self):
        self.on_redraw()

    def on_redraw(self):
        channels = np.unique([sv.source["channel"] for sv in self.source_view_registry])

        spec_results = {}
        for ch in channels:
            t1, t2 = self.view_state.get("current_range")
            t_arr, sig = self.state.get("sound_object").time_slice(t1, t2)
            sig -= np.mean(sig, axis=0)

            if self.state.get("lowres_preview"):
                # Scale down the resolution of the spectrogram computed
                t_spec, f_spec, spec, _ = spectrogram(
                    sig[:, ch],
                    self.state.get("sound_object").sampling_rate,
                    spec_sample_rate=(
                        read_default(self.settings, "SPEC_SAMPLE_RATE") /
                        read_default(self.settings, "LORES_SAMPLE_RATE_FACTOR")
                    ),
                    freq_spacing=(
                        read_default(self.settings, "SPEC_FREQ_SPACING") *
                        read_default(self.settings, "LORES_FREQ_SPACING_FACTOR")
                    ),
                    min_freq=read_default(self.settings, "MIN_FREQ"),
                    max_freq=read_default(self.settings, "MAX_FREQ"),
                )
                spec = np.repeat(spec, read_default(self.settings, "LORES_FREQ_SPACING_FACTOR"), axis=0)
                spec = np.repeat(spec, read_default(self.settings, "LORES_SAMPLE_RATE_FACTOR"), axis=1)
            else:
                t_spec, f_spec, spec, _ = spectrogram(
                    sig[:, ch],
                    self.state.get("sound_object").sampling_rate,
                    spec_sample_rate=read_default(self.settings, "SPEC_SAMPLE_RATE"),
                    freq_spacing=read_default(self.settings, "SPEC_FREQ_SPACING"),
                    min_freq=read_default(self.settings, "MIN_FREQ"),
                    max_freq=read_default(self.settings, "MAX_FREQ"),
                )

            spec_results[ch] = (t_spec, f_spec, spec)

        for source_view in self.source_view_registry:
            source_view.draw_spectrogram(*spec_results[source_view.source["channel"]])
            source_view._clear_drag_lines()
            source_view._clear_vertical_lines()


class SourceView(widgets.QWidget):
    """Visualize the current interval for a given source (aka 1 unique vocalizer)

    Draws the spectrogram for the given interval
    """
    def __init__(self, source_idx=None, parent=None, events=None):
        super().__init__(parent)
        self.events = events
        self.state = AppState()
        self.view_state = ViewState()
        self.settings = QSettings("Theuniseen Lab", "Sound Separation")

        self.source_idx = source_idx
        self.source = self.state.get("sources")[self.source_idx]

        self.init_ui()

        self.events.rangeSelected.connect(self.on_range_selected)
        # self.events.rangeChanged.connect(self.on_range_changed)

    def init_ui(self):
        self.drag_curves = {}

        self.spectrogram_viewbox = SpectrogramViewBox()
        self.spectrogram_plot = pg.PlotWidget(viewBox=self.spectrogram_viewbox)
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image)

        self.spectrogram_plot.selected_range_line_start = pg.InfiniteLine()
        self.spectrogram_plot.selected_range_line_stop = pg.InfiniteLine()
        self.spectrogram_plot.addItem(self.spectrogram_plot.selected_range_line_start)
        self.spectrogram_plot.addItem(self.spectrogram_plot.selected_range_line_stop)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

        self.spectrogram_viewbox.dragComplete.connect(self.on_drag_complete)
        self.spectrogram_viewbox.dragInProgress.connect(self.on_drag_in_progress)
        self.spectrogram_viewbox.clicked.connect(self.on_click)
        # self.spectrogram_viewbox.menuSelection.connect(self.on_menu_selection)

    def draw_spectrogram(self, t_spec, f_spec, spec):
        # Do the db conversion
        logspec = 20 * np.log10(np.abs(spec))
        max_b = logspec.max()
        min_b = logspec.max() - 40
        logspec[logspec < min_b] = min_b
        self.image.setImage(logspec.T)

    def _get_drag_mode(self, click_location):
        """Use the click location to determine how the dragging will affect
        the selected range

        "move": moves the entire range
        "left": moves only the left boundary
        "right": moves only the right boundary
        "new": creates a new range
        """
        if not self.view_state.has("selected_range"):
            return "new", None

        elif self.view_state.has("selected_range"):
            current_range = self.view_state.get("selected_range")
            base_t = self.view_state.get("current_range")[0]
            start_t, end_t = current_range
            spec_sample_rate = read_default(self.settings, "SPEC_SAMPLE_RATE")
            start_idx = int(round((start_t - base_t) * spec_sample_rate))
            end_idx = int(round((end_t - base_t) * spec_sample_rate))
            buffer = (end_idx - start_idx) / 10

            if start_idx + buffer <= click_location.x() <= end_idx - buffer:
                ## This would normally be "move", (start_idx, end_idx) but I
                ## don't wnat to enable click and drag of intervals
                return "new", None
            elif np.abs(start_idx - click_location.x()) < buffer:
                return "left", start_idx
            elif np.abs(end_idx - click_location.x()) < buffer:
                return "right", end_idx
            else:
                return "new", None

    def _clear_drag_lines(self):
        if len(self.drag_curves):
            self.spectrogram_plot.removeItem(self.drag_curves[self.spectrogram_plot])

    def _clear_vertical_lines(self):
        self.spectrogram_plot.selected_range_line_start.setVisible(False)
        self.spectrogram_plot.selected_range_line_stop.setVisible(False)

    def _draw_drag_curves(self, plot, start, end):
        pen = pg.mkPen((59, 124, 32, 255))
        self._clear_drag_lines()
        curve = pg.PlotCurveItem(
            [start.x(), end.x()],
            [start.y(), end.y()],
        )
        plot.addItem(curve)
        return curve

    def on_drag_in_progress(self, start, end):
        drag_mode, extra = self._get_drag_mode(start)

        dx = end.x() - start.x()
        self._clear_drag_lines()
        if drag_mode == "move":
            start_idx, end_idx = extra
            self.spectrogram_plot.selected_range_line_start.setValue(start_idx + dx)
            self.spectrogram_plot.selected_range_line_stop.setValue(end_idx + dx)
        elif drag_mode == "left":
            start_idx = extra
            self.spectrogram_plot.selected_range_line_start.setValue(start_idx + dx)
        elif drag_mode == "right":
            end_idx = extra
            self.spectrogram_plot.selected_range_line_stop.setValue(end_idx + dx)
        elif drag_mode == "new":
            curve_1 = self._draw_drag_curves(self.spectrogram_plot, start, end)
            self.drag_curves = {
                self.spectrogram_plot: curve_1,
            }

    def on_drag_complete(self, start, end):
        dx = 0

        drag_mode, extra = self._get_drag_mode(start)

        spec_sample_rate = read_default(self.settings, "SPEC_SAMPLE_RATE")
        if drag_mode == "new":
            start_dt = start.x() / spec_sample_rate
            end_dt = end.x() / spec_sample_rate
            base_t = self.view_state.get("current_range")[0]
            start_t = base_t + start_dt
            end_t = base_t + end_dt
        else:
            start_t, end_t = self.view_state.get("selected_range")
            base_t = self.view_state.get("current_range")[0]
            dx = end.x() - start.x()
            if drag_mode == "left":
                start_idx = extra
                start_dt = (start_idx + dx) / spec_sample_rate
                new_start_t = base_t + start_dt
                start_t, end_t = min(new_start_t, end_t), max(new_start_t, end_t)
            elif drag_mode == "right":
                end_idx = extra
                end_dt = (end_idx + dx) / spec_sample_rate
                new_end_t = base_t + end_dt
                start_t, end_t = min(start_t, new_end_t), max(start_t, new_end_t)
            elif drag_mode == "move":
                start_idx, end_idx = extra
                start_dt = (start_idx + dx) / spec_sample_rate
                end_dt = (end_idx + dx) / spec_sample_rate
                start_t = base_t + start_dt
                end_t = base_t + end_dt

        self.events.rangeSelected.emit(start_t, end_t)

    def on_click(self, loc):
        """Process a click event
        """
        self.events.rangeSelected.emit(None, None)

    def on_range_selected(self, start_t, end_t):
        spec_sample_rate = read_default(self.settings, "SPEC_SAMPLE_RATE")

        if start_t is None or end_t is None:
            self._clear_drag_lines()
            if self.view_state.has("selected_range"):
                self.view_state.clear("selected_range")
            self.spectrogram_plot.selected_range_line_start.setVisible(False)
            self.spectrogram_plot.selected_range_line_stop.setVisible(False)
        else:
            base_t = self.view_state.get("current_range")[0]
            start_t, end_t = min(start_t, end_t), max(start_t, end_t)
            self.view_state.set("selected_range", (start_t, end_t))
            start_idx = int(round((start_t - base_t) * spec_sample_rate))
            end_idx = int(round((end_t - base_t) * spec_sample_rate))
            self.spectrogram_plot.selected_range_line_start.setVisible(True)
            self.spectrogram_plot.selected_range_line_stop.setVisible(True)
            self.spectrogram_plot.selected_range_line_start.setValue(start_idx)
            self.spectrogram_plot.selected_range_line_stop.setValue(end_idx)

























pass
