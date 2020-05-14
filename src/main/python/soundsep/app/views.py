from functools import partial

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

from detection.thresholding import threshold_all_events

from app.components import SpectrogramViewBox
from app.state import AppState, ViewState
from app.settings import fonts, read_default
from app.style import qss
from app.utils import TimeScrollManager


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
        self.events.setBusy.connect(self.on_set_busy)

    def init_ui(self):
        self.topBar = widgets.QHBoxLayout()
        self.topBarLabel = widgets.QLabel("")
        self.topBarLabel.setDisabled(True)
        self.topBar.addWidget(self.topBarLabel)
        self.topBar.addStretch(1)

        self.busyIndicator = widgets.QLabel()
        busyGif = gui.QMovie("images/loading_icon.gif")
        busyGif.setScaledSize(QtCore.QSize(16, 16))
        self.busyIndicator.setMovie(busyGif)
        self.busyIndicator.setAlignment(QtCore.Qt.AlignRight)
        busyGif.start()

        self.topLeftBox = widgets.QGroupBox("Sources")
        self.topRightBox = widgets.QGroupBox("Sound Viewer")
        self.bottomBox = widgets.QGroupBox("Control Panel")

        self.topSplitter = widgets.QSplitter(Qt.Horizontal)
        self.topSplitter.addWidget(self.topLeftBox)
        self.topSplitter.addWidget(self.topRightBox)
        self.topSplitter.setStretchFactor(0, 0)
        self.topSplitter.setStretchFactor(1, 1)

        self.tab_panel = widgets.QTabWidget(self)
        bottomBoxLayout = widgets.QHBoxLayout()
        bottomBoxLayout.addWidget(self.tab_panel)
        self.bottomBox.setLayout(bottomBoxLayout)

        self.mainLayout = widgets.QGridLayout()
        self.mainLayout.addLayout(self.topBar, 0, 0)
        self.mainLayout.addWidget(self.busyIndicator, 0, 5, 1, 1)

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

        self.vocal_periods_view = VocalPeriodsView(
            None,
            events=self.events,
        )
        self.tab_panel.addTab(self.vocal_periods_view, "Vocal Periods")

    def on_set_busy(self, busy):
        self.busyIndicator.setVisible(busy)

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
            "hidden": False
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
            "channel": channel,
            "hidden": False
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
        eyeIcon = gui.QIcon("images/eye_icon.svg")
        self.hideButton = widgets.QToolButton(self)
        self.hideButton.setCheckable(True)
        if self.source["hidden"]:
            self.hideButton.toggle()
        self.hideButton.setIcon(eyeIcon)

        self.nameInput.returnPressed.connect(self.editButton.click)
        self.editButton.clicked.connect(self.toggle_edit_mode)
        self.deleteButton.clicked.connect(self.delete_source)
        self.hideButton.clicked.connect(self.hide_source)

        self.mainLayout.addWidget(self.nameInput)
        self.mainLayout.addWidget(self.channelInput)
        self.mainLayout.addWidget(self.editButton)
        self.mainLayout.addWidget(self.hideButton)
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

    def hide_source(self):
        if self.hideButton.isChecked() and self.source["hidden"] == False:
            self.source["hidden"] = True
            self.events.sourcesChanged.emit()
        elif not self.hideButton.isChecked() and self.source["hidden"] == True:
            self.source["hidden"] = False
            self.events.sourcesChanged.emit()


class AudioView(widgets.QWidget):

    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.events = events
        self.state = AppState()
        self.view_state = ViewState()
        self.settings = QSettings("Theuniseen Lab", "Sound Separation")
        self.timescroll_manager = TimeScrollManager(None)

        self.init_ui()

        # Timer to turn on/off low resolution views.
        # When the timer goes off, redraw in high resolution
        # Set with set_lowres(secs)
        self._lowres_timer = QtCore.QTimer()

        self.state.set("lowres_preview", False)
        self.source_view_registry = {}

        self.events.setPosition[object].connect(self.on_set_position)
        self.events.setPosition[object, object].connect(self.on_set_position)
        self.events.sourcesChanged.connect(self.on_sources_changed)
        self.events.rangeChanged.connect(self.on_range_changed)
        self.events.dataLoaded.connect(self.on_data_loaded)
        self.events.zoomEvent[int].connect(self.on_zoom)
        self.events.zoomEvent[int, float].connect(self.on_zoom)
        self._lowres_timer.timeout.connect(self.on_lowres_timer)

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

        self.windowInfoLayout = widgets.QHBoxLayout()
        self.time_label = widgets.QLabel()
        self.windowInfoLayout.addWidget(self.time_label)
        self.windowInfoLayout.addStretch()

        self.mainLayout.addLayout(self.topBarLayout)
        self.mainLayout.addLayout(self.currentSourcesLayout)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.scrollbar)
        self.mainLayout.addLayout(self.windowInfoLayout)

        self.setLayout(self.mainLayout)

    def _update_scroll_bar(self):
        steps = self.timescroll_manager.pages()
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(steps[-1] - 1)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setPageStep(self.timescroll_manager.page_step)

    def _update_time_label(self):
        t1, t2 = self.view_state.get("current_range")
        self.time_label.setText("{:.2f}s - {:.2f}s".format(t1, t2))

    def on_lowres_timer(self):
        self.on_redraw(lowres_timeout=0)

    def on_zoom(self, direction, pos=None):
        """Adjust the window size"""
        if pos:
            time_center = pos
            frac = (pos - self.view_state.get("current_range")[0]) / read_default.WINDOW_SIZE
        else:
            time_center = np.mean(self.view_state.get("current_range"))

        if direction < 0 and read_default.WINDOW_SIZE < read_default.MAX_WINDOW_SIZE:
            read_default.set(
                "WINDOW_SIZE",
                min(read_default.MAX_WINDOW_SIZE, 1.1 * read_default.WINDOW_SIZE)
            )
        elif direction > 0 and read_default.WINDOW_SIZE > read_default.MIN_WINDOW_SIZE:
            read_default.set(
                "WINDOW_SIZE",
                max(read_default.MIN_WINDOW_SIZE, read_default.WINDOW_SIZE / 1.1)
            )

        steps = self.timescroll_manager.pages()
        self.scrollbar.setMaximum(steps[-1] - 1)

        for sv in self.source_view_registry:
            sv._set_spectrogram_plot_limits()
            sv._draw_xaxis()

        if pos:
            self.events.setPosition[object, object].emit(time_center, frac)
        else:
            self.events.setPosition[object, object].emit(time_center, TimeScrollManager.ALIGN_CENTER)

    def on_slider_press(self):
        self.state.set("lowres_preview", True)

    def on_slider_release(self):
        self.state.set("lowres_preview", False)
        self.on_scrollbar_value_change(self.scrollbar.value())

    def set_page(self, page):
        """Generic way update the range and emit the rangeChanged event
        when you might not have actually changed the scroll state.
        """
        if page != self.scrollbar.value():
            self.scrollbar.setValue(page)
        else:
            self.on_scrollbar_value_change(page)

    def on_scrollbar_value_change(self, new_value):
        t1, t2 = self.timescroll_manager.page2time(new_value)
        self.view_state.set("current_range", (t1, t2))
        self.events.rangeChanged.emit()

    def on_data_loaded(self):
        self.timescroll_manager.set_max_time(self.state.get("sound_object").t_max)
        self._update_scroll_bar()
        self.on_scrollbar_value_change(0)
        self.on_sources_changed()

    def on_set_position(self, t, align=None):
        if align:
            new_page = self.timescroll_manager.time2page(t, align=align)
        else:
            new_page = self.timescroll_manager.time2page(t)

        self.set_page(new_page)

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
            source_view.setVisible(not source["hidden"])

        self.on_redraw()

    def on_range_changed(self):
        self._update_time_label()
        self.on_redraw()

    def on_redraw(self, lowres_timeout=1000):
        self._lowres_timer.stop()
        channels = np.unique([sv.source["channel"] for sv in self.source_view_registry])

        spec_results = {}
        for ch in channels:
            t1, t2 = self.view_state.get("current_range")
            t_arr, sig = self.state.get("sound_object").time_slice(t1, t2)
            sig -= np.mean(sig, axis=0)

            if lowres_timeout:
                # Scale down the resolution of the spectrogram computed
                sr_scale = 1
                fs_scale = 1
                for i, scale in enumerate(read_default.LORES_SCALES):
                    if read_default.WINDOW_SIZE > scale:
                        sr_scale = read_default.LORES_SAMPLE_RATE_FACTOR[i]
                        fs_scale = read_default.LORES_FREQ_SPACING_FACTOR[i]
                t_spec, f_spec, spec, _ = spectrogram(
                    sig[:, ch],
                    self.state.get("sound_object").sampling_rate,
                    spec_sample_rate=(
                        read_default.SPEC_SAMPLE_RATE /
                        sr_scale
                    ),
                    freq_spacing=(
                        read_default.SPEC_FREQ_SPACING *
                        fs_scale
                    ),
                    min_freq=read_default.MIN_FREQ,
                    max_freq=read_default.MAX_FREQ,
                )
                spec = np.repeat(spec, sr_scale, axis=1)
                spec = np.repeat(spec, fs_scale, axis=0)

                self._lowres_timer.start(lowres_timeout)
                self.events.setBusy.emit(True)
            else:
                t_spec, f_spec, spec, _ = spectrogram(
                    sig[:, ch],
                    self.state.get("sound_object").sampling_rate,
                    spec_sample_rate=read_default.SPEC_SAMPLE_RATE,
                    freq_spacing=read_default.SPEC_FREQ_SPACING,
                    min_freq=read_default.MIN_FREQ,
                    max_freq=read_default.MAX_FREQ,
                )
                self.events.setBusy.emit(False)

            spec_results[ch] = (t_spec, f_spec, spec)

        for source_view in self.source_view_registry:
            source_view.draw_spectrogram(*spec_results[source_view.source["channel"]])
            source_view._clear_drag_lines()
            source_view._update_highlighted_range()
            source_view._clear_vertical_lines()
            source_view._draw_xaxis()

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
        self.events.rangeHighlighted.connect(self.on_range_highlighted)

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

        self.spectrogram_plot.highlighted_range_line_start = pg.InfiniteLine(pen=pg.mkPen("g", width=1))
        self.spectrogram_plot.highlighted_range_line_stop = pg.InfiniteLine(pen=pg.mkPen("g", width=1))
        self.spectrogram_plot.addItem(self.spectrogram_plot.highlighted_range_line_start)
        self.spectrogram_plot.addItem(self.spectrogram_plot.highlighted_range_line_stop)

        self.spectrogram_plot.hideAxis("left")
        self.spectrogram_plot.hideButtons()  # Gets rid of "A" autorange button
        self._set_spectrogram_plot_limits()

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

        self.spectrogram_viewbox.dragComplete.connect(self.on_drag_complete)
        self.spectrogram_viewbox.dragInProgress.connect(self.on_drag_in_progress)
        self.spectrogram_viewbox.clicked.connect(self.on_click)
        self.spectrogram_viewbox.zoomEvent.connect(self.on_zoom)
        # self.spectrogram_viewbox.menuSelection.connect(self.on_menu_selection)

    def _set_spectrogram_plot_limits(self):
        ymax = int(
            (read_default.MAX_FREQ - read_default.MIN_FREQ) /
            read_default.SPEC_FREQ_SPACING
        )
        xmax = int(read_default.WINDOW_SIZE * read_default.SPEC_SAMPLE_RATE)
        self.spectrogram_plot.getViewBox().setRange(
            xRange=(0, xmax),
            yRange=(0, ymax),
            padding=0,
            disableAutoRange=True
        )

    def _nice_tick_spacing(self, win_size):
        """Compute a nice tick spacing for the given window size
        """
        if win_size < 60:
            first_guess = win_size / 10

        choices = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_choice = np.searchsorted(choices, first_guess)
        return choices[best_choice]

    def _draw_xaxis(self):
        ax_spec = self.spectrogram_plot.getAxis("bottom")

        win_size = read_default.WINDOW_SIZE
        base_t = self.view_state.get("current_range")[0]

        ticks = []
        spacing = self._nice_tick_spacing(win_size)
        offset = base_t % spacing
        for t in np.arange(-offset, win_size - offset, spacing):
            # Choose ticks at the nearest multiples of spacing
            samples = int(np.round(t * read_default.SPEC_SAMPLE_RATE))
            ticks.append([samples, np.around(base_t + t, 2)])

        ax_spec.setTicks([ticks])

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
            spec_sample_rate = read_default.SPEC_SAMPLE_RATE
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

    def on_zoom(self, direction, pos):
        """Pass the zoom signal up to the AudioView

        Trieds to keep the cursor position fixed.
        """
        t_start, t_stop = self.view_state.get("current_range")
        t_cursor = t_start + pos[0] * (t_stop - t_start)

        if direction > 0:
            self.events.zoomEvent[int, float].emit(1, t_cursor)
        elif direction < 0:
            self.events.zoomEvent[int, float].emit(-1, t_cursor)

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

        spec_sample_rate = read_default.SPEC_SAMPLE_RATE
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
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

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

    def on_range_highlighted(self, start_t, end_t):
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        if start_t is None or end_t is None:
            # pass
            self._clear_drag_lines()
            if self.view_state.has("highlighted_range"):
                self.view_state.clear("highlighted_range")
        else:
            start_t, end_t = min(start_t, end_t), max(start_t, end_t)
            self.view_state.set("highlighted_range", (start_t, end_t))
            self._update_highlighted_range()

    def _update_highlighted_range(self):
        base_t = self.view_state.get("current_range")[0]
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        if self.view_state.has("highlighted_range"):
            start_t, end_t = self.view_state.get("highlighted_range")
            start_idx = int(round((start_t - base_t) * spec_sample_rate))
            end_idx = int(round((end_t - base_t) * spec_sample_rate))

            self.spectrogram_plot.highlighted_range_line_start.setVisible(True)
            self.spectrogram_plot.highlighted_range_line_stop.setVisible(True)
            self.spectrogram_plot.highlighted_range_line_start.setValue(start_idx)
            self.spectrogram_plot.highlighted_range_line_stop.setValue(end_idx)
        else:
            self.spectrogram_plot.highlighted_range_line_start.setVisible(False)
            self.spectrogram_plot.highlighted_range_line_stop.setVisible(False)


class VocalPeriodsView(widgets.QScrollArea):

    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.state = AppState()
        self.view_state = ViewState()
        self.events = events

        self.init_ui()

    def init_ui(self):
        self.mainLayout = widgets.QHBoxLayout()

        self.vocalPeriodsLayout = widgets.QHBoxLayout()

        self.detectButton = widgets.QPushButton("Detect Potential Vocal Periods")
        self.mainLayout.addWidget(self.detectButton)
        self.mainLayout.addLayout(self.vocalPeriodsLayout)
        self.mainLayout.addStretch()

        self.setLayout(self.mainLayout)

        self.detectButton.clicked.connect(self.on_detect)

    def on_detect(self):
        # Detect on all channels with sources - merge them, then present them as options

        audio_signal = self.state.get("sound_object")
        first_source = self.state.get("sources")[0]
        events = threshold_all_events(
                audio_signal,
                window_size=10.0,
                channel=first_source["channel"],
                t_start=None,
                t_stop=None,
                ignore_width=0.01,
                min_size=0.01,
                fuse_duration=2.0,
                threshold_z=2.0,
                amp_env_mode="broadband"
        )

        first_source["vocal_periods"] = events

        for i in reversed(range(self.vocalPeriodsLayout.count())):
            item = self.vocalPeriodsLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        for event in events:
            button = widgets.QPushButton("{:.2f}\nto\n{:.2f}".format(event[0], event[1]))
            self.vocalPeriodsLayout.addWidget(button)
            button.clicked.connect(partial(self.on_event_clicked, event))

    def on_event_clicked(self, event):
        t1, t2 = event
        t_center = np.mean([t1, t2])
        self.events.setPosition[object, object].emit(t_center, TimeScrollManager.ALIGN_CENTER)
        self.view_state.set("highlighted_range", (t1, t2))
        self.events.rangeHighlighted.emit(t1, t2)

















pass
