import re
import uuid
from contextlib import contextmanager
from functools import partial

import pandas as pd
import pyqtgraph as pg
import numpy as np
from scipy.signal import savgol_filter
import sounddevice as sd
from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets

from soundsig.sound import spectrogram

from audio_utils import get_amplitude_envelope
from detection.thresholding import threshold_all_events, threshold_events, fuse_events

from app.components import SpectrogramViewBox
from app.context import context
from app.state import AppState, ViewState
from app.settings import fonts, read_default
from app.style import qss
from app.utils import TimeScrollManager, ThresholdAdjuster
from app.workers import SpectrogramWorker, VocalPeriodsWorker


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

        self.tab_panel = widgets.QTabWidget(self)
        bottomBoxLayout = widgets.QHBoxLayout()
        bottomBoxLayout.addWidget(self.tab_panel)
        self.bottomBox.setLayout(bottomBoxLayout)

        self.mainLayout = widgets.QGridLayout()
        self.mainLayout.addLayout(self.topBar, 0, 0)

        self.mainLayout.addWidget(self.topSplitter, 1, 0, 1, 6)
        # self.mainLayout.addWidget(self.bottomBox, 2, 0, 1, 6)

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

    def on_data_load(self):
        self.topLeftBox.setDisabled(False)
        self.topRightBox.setDisabled(False)
        self.topBarLabel.setText(self.state.get("sound_file"))


class SourceManagerView(widgets.QScrollArea):
    """List view of sources and editing options

    A "Source" is a sound source in the current study. For example, one
    vocalizing individual. A source is associated with a channel. Multiple
    sources can be associated with the same channel, it is up to the user
    to distinguish them.
    """
    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.state = AppState()
        self.events = events

        self.init_ui()

        self.events.sourcesChanged.connect(self.on_sources_changed)
        self.events.dataLoaded.connect(self.on_sources_changed)
        self.events.createSource.connect(self.on_create_source)

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()

        self.currentSourcesLayout = widgets.QVBoxLayout()
        self.addSourceButton = widgets.QToolButton()
        addIcon = gui.QIcon(context.get_resource("images/plus_icon.svg"))
        self.addSourceButton = widgets.QToolButton(self)
        self.addSourceButton.setIcon(addIcon)
        self.addSourceButton.setToolTip("Create a new source")
        self.addSourceButton.setFixedWidth(50)
        self.addSourceButton.clicked.connect(self.add_source)

        self.mainLayout.addLayout(self.currentSourcesLayout)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.addSourceButton)
        self.setLayout(self.mainLayout)

    def add_source(self):
        """Create a new source with a blank title in edit mode and incremented
        to a channel that isn't used yet (if any)
        """
        current_channels = [s["channel"] for s in self.state.get("sources")]
        for ch in range(self.state.get("sound_object").n_channels):
            if ch not in current_channels:
                new_channel = ch
                break
        else:
            new_channel = 0

        self.events.createSource.emit({
            "name": "new{}".format(len(current_channels) + 1),
            "channel": new_channel,
            "hidden": False
        })

    def on_sources_changed(self):
        """Repopulate the list with the current sources
        """
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

    def on_create_source(self, source_dict):
        """
        This is a separate function from add_source() because originally
        add_source opened a separate window and needed to trigger
        the actual creation of the source to emit sourcesChanged. This could
        be consolidated...
        """
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


class SourceEditorView(widgets.QWidget):
    """Display of source information and options to edit/delete/hide/show

    One of these is created for each source and created in the
    SourceManagerView
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
        self.nameInput.setStyleSheet("QLineEdit {qproperty-cursorPosition: 0;}");
        self.channelInput = widgets.QComboBox(self)
        for ch in range(self.state.get("sound_object").n_channels):
            self.channelInput.addItem(str(ch), ch)
        self.channelInput.setCurrentIndex(self.source["channel"])
        self.nameInput.setDisabled(True)
        self.channelInput.setDisabled(True)

        editIcon = gui.QIcon(context.get_resource("images/edit_icon.svg"))
        self.editButton = widgets.QToolButton(self)
        self.editButton.setIcon(editIcon)
        self.editButton.setToolTip("Edit source name/channel")
        self.editButton.setCheckable(True)
        deleteIcon = gui.QIcon(context.get_resource("images/delete_icon.svg"))
        self.deleteButton = widgets.QToolButton(self)
        self.deleteButton.setIcon(deleteIcon)
        eyeIcon = gui.QIcon(context.get_resource("images/eye_icon.svg"))
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
            self.editButton.setToolTip("Submit changes [Return]")
            self.nameInput.setDisabled(False)
            self.channelInput.setDisabled(False)
        else:
            self.editButton.setToolTip("Edit source name/channel")
            if self.nameInput.text() == "":
                widgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Cannot leave source name blank",
                )
                return

            self.nameInput.setDisabled(True)
            self.channelInput.setDisabled(True)
            self.repaint()
            new_name = self.nameInput.text()
            new_channel = self.channelInput.currentIndex()
            if (self.source["name"] == new_name and
                    self.source["channel"] == new_channel):
                # no changes were made
                return
            else:
                self.source["name"] = new_name or self.source["name"]
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
        self.settings = QSettings("Theunissen Lab", "Sound Separation")
        self.timescroll_manager = TimeScrollManager(None)
        self._redrawing = False

        self.init_ui()

        # Timer to turn on/off low resolution views.
        # When the timer goes off, redraw in high resolution
        # Set with set_lowres(secs)
        self._last_spec_key = None
        self._lowres_timer = QtCore.QTimer()
        self.thread = None

        self.source_view_registry = {}

        self.events.setPosition[object].connect(self.on_set_position)
        self.events.setPosition[object, object].connect(self.on_set_position)
        self.events.sourcesChanged.connect(self.on_sources_changed)
        self.events.rangeChanged.connect(self.on_range_changed)
        self.events.dataLoaded.connect(self.on_data_loaded)
        self.events.zoomEvent[int].connect(self.on_zoom)
        self.events.zoomEvent[int, float].connect(self.on_zoom)
        self.events.zoomEvent[str].connect(self.on_zoom)
        self.events.triggerShortcut.connect(self.on_shortcut)
        self.events.playAudio.connect(self.on_play_audio)

        # Timer for having a delay on computing the full spectrogram.
        # If the user is scrolling through time, don't want to compute
        # the full resolution spectrogram constantly.
        self._lowres_timer.timeout.connect(self.on_lowres_timer)

    def init_ui(self):
        self.mainLayout = widgets.QVBoxLayout()

        self.topBarLayout = widgets.QHBoxLayout()

        self.showAmpenvToggle = widgets.QPushButton("Show Ampenv")
        self.showAmpenvToggle.setCheckable(True)
        self.selectByThresholdToggle = widgets.QPushButton("Select by Threshold")
        self.selectByThresholdToggle.setCheckable(True)
        self.autoSearchToggle = widgets.QPushButton("Auto Search on Drag")
        self.autoSearchToggle.setCheckable(True)

        if self.state.has("autosearch"):
            self.autoSearchToggle.toggle()

        self.topBarLayout.addWidget(self.showAmpenvToggle)
        self.topBarLayout.addWidget(self.selectByThresholdToggle)
        self.topBarLayout.addWidget(self.autoSearchToggle)

        self.topBarLayout.addStretch()
        self.showAmpenvToggle.clicked.connect(self.on_show_ampenv)
        self.selectByThresholdToggle.clicked.connect(self.on_toggle_threshold)
        self.autoSearchToggle.clicked.connect(self.on_auto_search)

        self.ampenvScaleSlider = widgets.QSlider()
        self.ampenvScaleSlider.setTickPosition(widgets.QSlider.TicksBothSides)
        self.ampenvScaleSlider.setVisible(False)
        self.ampenvScaleSlider.setMinimum(1)
        self.ampenvScaleSlider.setMaximum(20)
        self.ampenvScaleSlider.setValue(11)
        self.ampenvScaleSlider.setSingleStep(1)
        self.ampenvScaleSlider.valueChanged.connect(self.on_ampenv_scale_changed)

        self.currentSourcesLayout = widgets.QVBoxLayout()

        self.currentSourcesPanel = widgets.QHBoxLayout()
        self.currentSourcesPanel.addLayout(self.currentSourcesLayout)
        self.currentSourcesPanel.addWidget(self.ampenvScaleSlider)

        self.scrollbar = widgets.QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(100)
        self.timeInput = widgets.QLineEdit(self)
        self.timeInput.setPlaceholderText("Jump to...")

        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_change)
        self.timeInput.returnPressed.connect(self.on_timeinput_value_change)
        self.timeInput.textChanged.connect(self.timeinput_live_cleaning)

        self.windowInfoLayout = widgets.QHBoxLayout()
        self.time_label = widgets.QLabel()
        self.windowInfoLayout.addWidget(self.timeInput)
        self.windowInfoLayout.addStretch()
        self.windowInfoLayout.addWidget(self.time_label)

        self.mainLayout.addLayout(self.topBarLayout)
        self.mainLayout.addLayout(self.currentSourcesPanel)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.scrollbar)
        self.mainLayout.addWidget
        self.mainLayout.addLayout(self.windowInfoLayout)

        self.setLayout(self.mainLayout)

    @contextmanager
    def redraw_context(self, *args, **kwargs):
        # Context manager so that multiple redraws don't happen in the same loop
        # kind of janky
        if self._redrawing:
            draw = False
        else:
            self._redrawing = True
            draw = True

        try:
            yield
        finally:
            if draw:
                self.on_redraw(*args, **kwargs)
                self._redrawing = False

    def _update_scroll_bar(self):
        """Set the scrollbar parameters
        """
        steps = self.timescroll_manager.pages()
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(steps[-1] - 1)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setPageStep(self.timescroll_manager.page_step)

    def _update_time_label(self):
        t1, t2 = self.view_state.get("current_range")
        self.time_label.setText(
            "{:.2f}s - {:.2f}s (out of {:.2f}s)".format(
                t1, t2, self.timescroll_manager.max_time
            )
        )

    def _reset_thread(self):
        if self.thread is not None:
            self.thread.exit()
        self.thread = QThread(self)
        return self.thread

    def stop_audio_playback(self):
        sd.stop()
        for source_view in self.source_view_registry:
            source_view.reset_playback_line()

    def closeEvent(self):
        if self.thread is not None:
            self.thread.stop()

    def on_shortcut(self, shortcut):
        """Handle keyboard shortcuts"""
        if shortcut in ("A", "D"):
            # Shortcuts for moving forward and back in time.
            frac = read_default.KEYBOARD_TIME_STEP_FRACTION
            current_page = self.scrollbar.value()
            if shortcut == "A":
                new_page = max(
                    current_page - self.timescroll_manager.page_step / frac,
                    0
                )
            elif shortcut == "D":
                new_page = min(
                    current_page + self.timescroll_manager.page_step / frac,
                    self.timescroll_manager.pages()[-1]
                )
            self.scrollbar.setValue(new_page)
        elif shortcut == "E":
            # Shortcut for toggling amplitude envelope mode
            self.showAmpenvToggle.toggle()
            self.on_show_ampenv()
        elif shortcut == "M":
            # Shortcut for toggling autosearch mode
            self.autoSearchToggle.toggle()
            self.on_auto_search()
        elif shortcut == "Space":
            if not self.view_state.has("source_focus"):
                # This is the default behavior if one of the sources isn't selected
                # If it is selected, then the playback function is delegated
                # to the SourceView object in focus
                for source_idx, source_view in enumerate(self.source_view_registry):
                    if not source_view.source["hidden"]:
                        self.events.playAudio.emit(source_idx)
                        break
        elif shortcut == "Escape":
            self.stop_audio_playback()

    def on_play_audio(self, source_idx):
        self.stop_audio_playback()
        self.source_view_registry[source_idx].play_audio()

    def on_show_ampenv(self):
        with self.redraw_context():
            if self.showAmpenvToggle.isChecked():
                self.view_state.set("show_ampenv", True)
            else:
                self.view_state.clear("show_ampenv")

    def on_toggle_threshold(self):
        if self.selectByThresholdToggle.isChecked():
            self.view_state.set("select_by_threshold", True)
        else:
            self.view_state.clear("select_by_threshold")

    def on_auto_search(self):
        if self.autoSearchToggle.isChecked():
            self.state.set("autosearch", True)
        else:
            self.state.clear("autosearch")

    def on_lowres_timer(self):
        """Draw the spectrogram in full hi resolution"""
        with self.redraw_context(lowres_timeout=0):
            pass

    def on_zoom(self, direction, pos=None):
        """Adjust the window size and reposition window.

        direction: +1 or -1 for zoom in/out
        pos: x location (time) where the cursor is zooming
        """
        if pos:
            time_center = pos
            frac = (pos - self.view_state.get("current_range")[0]) / read_default.WINDOW_SIZE
        else:
            time_center = np.mean(self.view_state.get("current_range"))

        if direction in (-1, 1):
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
        elif direction in ("selection", "max"):
            if direction == "selection":
                if self.view_state.has("selected_range"):
                    t0, t1 = self.view_state.get("selected_range")
                    time_center = np.mean([t0, t1])
                    read_default.set(
                        "WINDOW_SIZE",
                        t1 - t0
                    )
                else:
                    return
            elif direction == "max":
                read_default.set(
                    "WINDOW_SIZE",
                    read_default.MAX_WINDOW_SIZE
                )

        steps = self.timescroll_manager.pages()
        self.scrollbar.setMaximum(steps[-1] - 1)

        for sv in self.source_view_registry:
            sv._set_spectrogram_plot_limits()
            sv._draw_xaxis()

        if pos:
            self.events.setPosition[object, object].emit(time_center, frac)
        else:
            self.events.setPosition[object, object].emit(
                time_center,
                TimeScrollManager.ALIGN_CENTER
            )

    def set_page(self, page):
        """Generic way update the range and emit the rangeChanged event
        when you might not have actually changed the scroll state.
        """
        with self.redraw_context():
            if page != self.scrollbar.value():
                self.scrollbar.setValue(page)
            else:
                self.on_scrollbar_value_change(page)

    def on_ampenv_scale_changed(self, new_value):
        with self.redraw_context():
            self.view_state.set("scale_ampenv", (21 - new_value) / 10)

    def timeinput_live_cleaning(self, t):
        """Prevent any non-digit, ., or : characters in the time jump box

        If there are more than 2 colons, truncate the string to everything
        before the third colon
        """
        t = "".join([c for c in t if re.match("[0-9]|\.|:", c)])
        if t.count(":") > 2:
            t = ":".join(t.split(":")[:3])
        self.timeInput.setText(t)

    def on_timeinput_value_change(self):
        """Logic to interpret a number string as a time in seconds

        Allows floats, two floats seperated by a ':' (i.e. mm:ss)
        and three floats separated by ':'s (i.e. hh:mm:ss)
        """
        txt = self.timeInput.text()
        t_blocks = [s for s in txt.split(":") if len(s)][:3]
        try:
            if len(t_blocks) == 1:
                s = float(t_blocks[0])
            elif len(t_blocks) == 2:
                m = float(t_blocks[0])
                s = float(t_blocks[1]) + m * 60
            elif len(t_blocks) == 3:
                h = float(t_blocks[0])
                m = float(t_blocks[1]) + h * 60
                s = float(t_blocks[2]) + m * 60
        except ValueError:
            widgets.QMessageBox.warning(
                self,
                "Error",
                "Invalid time {}.".format(txt),
            )
            return
        self.on_set_position(s)
        self.timeInput.setText("")
        self.timeInput.clearFocus()

    def on_scrollbar_value_change(self, new_value):
        t1, t2 = self.timescroll_manager.page2time(new_value)
        self.view_state.set("current_range", (t1, t2))
        # sends signal that view range changed
        self.events.rangeChanged.emit()

    def on_data_loaded(self):
        with self.redraw_context():
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
        with self.redraw_context():
            for i in reversed(range(self.currentSourcesLayout.count())):
                item = self.currentSourcesLayout.itemAt(i)
                if item.widget():
                    item.widget().deleteLater()

            self.source_view_registry = []
            for i, source in enumerate(self.state.get("sources")):
                source_view = SourceView(source_idx=i, events=self.events)
                self.currentSourcesLayout.addWidget(source_view)
                self.source_view_registry.append(source_view)
                source_view.setVisible(not source["hidden"])

    def on_range_changed(self):
        with self.redraw_context():
            self.stop_audio_playback()
            self._update_time_label()
            self.view_state.clear("selected_spec_range")
            self.view_state.clear("selected_threshold_line")
            self.view_state.clear("selected_range")

    def on_redraw(self, lowres_timeout=500):
        """Computes spectrogram values for the current viewable window

        If lowres_timeout is passed, render the low resolution version of the
        spectrograms (fast), while also kicking off a timer for
        lowres_timeout (ms). After the timeout, a background job for the high
        resolution spectrogram will be started.
        """
        self._lowres_timer.stop()
        channels = np.unique([
            sv.source["channel"] for sv in self.source_view_registry
            if sv.source["hidden"] == False
        ])

        spec_results = {}

        t1, t2 = self.view_state.get("current_range")
        t_arr, sig = self.state.get("sound_object").time_slice(t1, t2)
        sig -= np.mean(sig, axis=0)

        if self.view_state.has("show_ampenv") and len(self.state.get("sources")):
            self.ampenvScaleSlider.setVisible(self.view_state.get("show_ampenv"))
        else:
            self.ampenvScaleSlider.setVisible(False)

        if lowres_timeout:
            for ch in channels:
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
                    cmplx=False,
                )
                spec = np.repeat(spec, sr_scale, axis=1)
                spec = np.repeat(spec, fs_scale, axis=0)

                # i think these are probably incorrect
                # should get the real way to get these from soundsig.
                t_spec = np.linspace(
                    t1,
                    t2,
                    spec.shape[1]
                )
                f_spec = np.linspace(
                    read_default.MIN_FREQ,
                    read_default.MAX_FREQ,
                    spec.shape[0]
                )

                self._lowres_timer.start(lowres_timeout)
                spec_results[ch] = (t_spec, f_spec, spec)
            self.on_spectrogram_completed(lores=True, spec_results=spec_results)
        else:
            # Since the spectrogram computation is asynchronous, the user
            # may have moved on to another view by the time the spectrogram
            # comes back from its thread.
            # So when it comes back (in on_spectrogram_completed()) we
            # make sure that the job finished is the last one requested by
            # checking if the _last_spec_key matches what we are waiting for.
            # A better way to do this is if we had a way of stopping the
            # spectrogram worker in its tracks when a new one is requested
            # but I don't know how to do that.
            self._last_spec_key = uuid.uuid4().hex
            self.worker = SpectrogramWorker(
                self._last_spec_key,
                channels,
                sig,
                self.state.get("sound_object").sampling_rate,
                spec_sample_rate=read_default.SPEC_SAMPLE_RATE,
                freq_spacing=read_default.SPEC_FREQ_SPACING,
                min_freq=read_default.MIN_FREQ,
                max_freq=read_default.MAX_FREQ,
                cmplx=False
            )
            self.worker.finished.connect(partial(
                self.on_spectrogram_completed,
                False,
            ))
            self._reset_thread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.compute)
            self.thread.start()

    def on_spectrogram_completed(self, lores=False, key=None, spec_results=None):
        """Draws the given spectrogram data for all sources
        """
        if not lores and key != self._last_spec_key:
            return

        for source_view in self.source_view_registry:
            if source_view.source["hidden"] == True:
                continue
            # compute the value to normalize amp env to across channels
            if self.view_state.has("show_ampenv") and lores == False:
                max_values = []
                for k, v in spec_results.items():
                    max_values.append(np.max(np.sum(v[2], axis=0)))

                if self.view_state.has("scale_ampenv"):
                    scale_factor = self.view_state.get("scale_ampenv")
                else:
                    scale_factor = 1

                source_view.draw_spectrogram(
                    *spec_results[source_view.source["channel"]],
                    show_ampenv=True,
                    ampenv_norm=np.max(max_values) * 1.1 * scale_factor
                )
            else:
                source_view.draw_spectrogram(
                    *spec_results[source_view.source["channel"]],
                    show_ampenv=False,
                )
            source_view._clear_drag_lines()
            source_view._update_highlighted_range()
            source_view._draw_intervals()
            source_view._clear_vertical_lines()
            source_view._draw_xaxis()


class SourceView(widgets.QWidget):
    """Visualize audio for a given source (aka 1 unique vocalizer)

    Draws the spectrogram for the given interval and handles
    selection functions, keyboard shortcuts, and user functions pertaining
    to selection of audio sections and labeling vocal intervals.
    """
    def __init__(self, source_idx=None, parent=None, events=None):
        super().__init__(parent)
        self.events = events
        self.state = AppState()
        self.view_state = ViewState()
        self.settings = QSettings("Theunissen Lab", "Sound Separation")

        self.source_idx = source_idx
        self.source = self.state.get("sources")[self.source_idx]
        self._drawn_intervals = []

        # Set up playback line
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_playback_line)
        self._playback_line_pos = -1

        self.init_ui()

        self.events.rangeSelected.connect(self.on_range_selected)
        self.events.rangeHighlighted.connect(self.on_range_highlighted)
        self.events.triggerShortcut.connect(self.on_shortcut)

        self._highlight_rect = None
        self._drawn_intervals = []
        self.reset_playback_line(visible=False)

    def init_ui(self):
        self.drag_curves = {}

        source_label_layout = widgets.QHBoxLayout()
        source_label = widgets.QLabel(self.source["name"])
        source_label.setFont(fonts.subheading)
        play_button = widgets.QPushButton("Play")
        play_button.clicked.connect(partial(self.events.playAudio.emit, self.source_idx))

        source_label_layout = widgets.QHBoxLayout()
        source_label_layout.addWidget(source_label)
        source_label_layout.addStretch()
        source_label_layout.addWidget(play_button)

        self.spectrogram_viewbox = SpectrogramViewBox()
        self.spectrogram_plot = pg.PlotWidget(viewBox=self.spectrogram_viewbox)
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image)

        self.spectrogram_plot.selected_range_line_start = pg.InfiniteLine()
        self.spectrogram_plot.selected_range_line_stop = pg.InfiniteLine()
        self.spectrogram_plot.spec_range_line_start = pg.InfiniteLine(angle=0)
        self.spectrogram_plot.spec_range_line_stop = pg.InfiniteLine(angle=0)
        self.spectrogram_plot.playback_line = pg.InfiniteLine()

        self.spectrogram_plot.addItem(self.spectrogram_plot.playback_line)
        self.spectrogram_plot.addItem(self.spectrogram_plot.selected_range_line_start)
        self.spectrogram_plot.addItem(self.spectrogram_plot.selected_range_line_stop)
        self.spectrogram_plot.addItem(self.spectrogram_plot.spec_range_line_start)
        self.spectrogram_plot.addItem(self.spectrogram_plot.spec_range_line_stop)

        self.spectrogram_viewbox.menuSelection.connect(self.on_menu_selection)

        self.spectrogram_plot.hideAxis("left")
        self.spectrogram_plot.hideButtons()  # Gets rid of "A" autorange button
        self._set_spectrogram_plot_limits()

        self.ampenv_plot = pg.PlotCurveItem([], pen=pg.mkPen("g", width=1.2))
        self.spectrogram_viewbox.addItem(self.ampenv_plot)

        layout = widgets.QVBoxLayout()
        layout.addLayout(source_label_layout)
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

        self.spectrogram_viewbox.dragComplete.connect(self.on_drag_complete)
        self.spectrogram_viewbox.dragInProgress.connect(self.on_drag_in_progress)
        self.spectrogram_viewbox.clicked.connect(self.on_click)
        self.spectrogram_viewbox.zoomEvent.connect(self.on_zoom)

    def reset_playback_line(self, start_at=None, visible=False):
        self.timer.stop()
        self._playback_line_pos = start_at or -1
        self.spectrogram_plot.playback_line.setVisible(visible)
        if self.view_state.has("playback_range"):
            self.view_state.clear("playback_range")

    def advance_playback_line(self):
        """Step the playback line forward in time

        Reset the line when it reaches the end
        """
        t0, _ = self.view_state.get("current_range")

        if not self.view_state.has("playback_range"):
            max_playback_line_pos = int(
                read_default.SPEC_SAMPLE_RATE *
                read_default.WINDOW_SIZE
            )
        else:
            _, selection_t1 = self.view_state.get("playback_range")
            max_playback_line_pos = int(
                read_default.SPEC_SAMPLE_RATE *
                (selection_t1 - t0)
            )

        if self._playback_line_pos < max_playback_line_pos:
            self._playback_line_pos += read_default.PLAYBACK_LINE_STEP
            self.spectrogram_plot.playback_line.setValue(self._playback_line_pos)
        else:
            self.reset_playback_line(visible=False)

    def play_audio(self):
        """Play the sound thats in the currently selected window"""
        if self.view_state.has("selected_range"):
            t0, t1 = self.view_state.get("selected_range")
        else:
            t0, t1 = self.view_state.get("current_range")

        t_arr, sig = self.state.get("sound_object").time_slice(t0, t1)
        sig -= np.mean(sig, axis=0)
        start_idx = int(round((
            (t0 - self.view_state.get("current_range")[0]) *
            read_default.SPEC_SAMPLE_RATE
        )))
        self.reset_playback_line(start_at=start_idx, visible=True)
        self.view_state.set("playback_range", [t0, t1])
        sd.play(
            sig[:, self.source["channel"]],
            self.state.get("sound_object").sampling_rate,
            blocking=False
        )

        self.timer.start(
            read_default.PLAYBACK_LINE_STEP * 1000 /
            read_default.SPEC_SAMPLE_RATE
        )

    def _set_spectrogram_plot_limits(self):
        """
        TODO: there are a lot of axes conversions (spectrogram units, amp
        env units, and plot units)
        """
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
        """Compute a nice tick spacing for the given window size for x-axis
        """
        if win_size < 60:
            first_guess = win_size / 10

        choices = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_choice = np.searchsorted(choices, first_guess)
        return choices[best_choice]

    def _seconds_to_human_readable(self, seconds):
        """Converts raw time in seconds to human readable format (1h2m 3.3s)"""
        result = ""
        hours = int(seconds / (60 * 60))
        minutes = int((seconds - hours * (60 * 60)) / 60)
        seconds = seconds - (minutes * 60) - (hours * 60 * 60)
        if hours:
            result += "{:d}:".format(hours)
        if minutes:
            if hours:
                result += "{:d}:".format(minutes).zfill(2)
            else:
                result += "{:d}:".format(minutes)

        if minutes:
            result += "{:.1f}".format(seconds).zfill(4)
        else:
            result += "{:.1f}".format(seconds)
        return result

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
            ticks.append([samples, self._seconds_to_human_readable(np.around(base_t + t, 2))])

        ax_spec.setTicks([ticks])

    def _clear_drag_lines(self):
        if self.view_state.has("selected_threshold_line"):
            self.view_state.clear("selected_threshold_line")
        if self.view_state.has("selected_spec_range"):
            self.view_state.clear("selected_spec_range")
        if len(self.drag_curves):
            self.spectrogram_plot.removeItem(self.drag_curves[self.spectrogram_plot])

    def _clear_vertical_lines(self):
        self.spectrogram_plot.selected_range_line_start.setVisible(False)
        self.spectrogram_plot.selected_range_line_stop.setVisible(False)
        self.spectrogram_plot.spec_range_line_start.setVisible(False)
        self.spectrogram_plot.spec_range_line_stop.setVisible(False)

    def _draw_drag_curves(self, plot, start, end):
        pen = pg.mkPen((59, 124, 32, 255))
        self._clear_drag_lines()
        curve = pg.PlotCurveItem(
            [start.x(), end.x()],
            [start.y(), end.y()],
        )
        plot.addItem(curve)
        return curve

    def draw_spectrogram(self, t_spec, f_spec, spec,
            show_ampenv=False, ampenv_norm=None):
        """Set the spectrogram image data

        Computes spectrogram in dB. Uses spectrogram to compute
        an amp env if amp env visualization is requested.
        """
        # Do the db conversion
        logspec = 20 * np.log10(np.abs(spec))
        max_b = logspec.max()
        min_b = logspec.max() - 40
        logspec[logspec < min_b] = min_b
        self.image.setImage(logspec.T)
        self._current_spectrogram_yscale = read_default.SPEC_FREQ_SPACING
        # instead of computing a separate ampenv, just use the spectrogram power
        if show_ampenv:
            ampenv = np.sum(spec, axis=0)
            # Smooth the "ampenv" (which is just computed from the spectrogram)
            ampenv = savgol_filter(ampenv, 21, 7)
            self.draw_ampenv(t_spec, ampenv, norm=ampenv_norm)
        else:
            self.ampenv_plot.setData([])

    def draw_ampenv(self, t_ampenv, ampenv, norm=None):
        viewbox = self.spectrogram_plot.getPlotItem().getViewBox()
        ((_, _), (_, ymax)) = viewbox.viewRange()

        # normalize to ymax value
        ampenv = ymax * ampenv / (norm or np.max(ampenv))
        ampenv -= np.min(ampenv)
        self._current_ampenv_yscale = (norm or np.max(ampenv)) / ymax
        self.ampenv_plot.setData(ampenv)

    def on_shortcut(self, shortcut):
        if not self.view_state.has("source_focus"):
            return

        if self.view_state.get("source_focus") != self.source_idx:
            return

        if shortcut == "W":
            self.on_find_in_selection(0)
        elif shortcut == "Shift+W":
            self.on_find_in_selection(1)
        elif shortcut == "X":
            self.on_delete_selection()
        elif shortcut == "S":
            self.on_find_in_selection(0)
        elif shortcut == "Shift+S":
            self.on_find_in_selection(-1)
        elif shortcut == "Q":
            # merge
            self.on_merge_selection()
        elif shortcut == "Z":
            self.on_find_in_selection(0, mode="max_zscore")
        elif shortcut == "Space":
            self.events.playAudio.emit(self.source_idx)

    def on_menu_selection(self, item, pos=None):
        if item == "search":
            self.on_find_in_selection(1)
        elif item == "merge":
            self.on_merge_selection()
        elif item == "delete":
            self.on_delete_selection()
        elif item == "desperate":
            self.on_find_in_selection(0, mode="max_zscore")
        elif item == "zoom_in":
            self.events.zoomEvent[str].emit("selection")
        elif item == "zoom_out":
            self.events.zoomEvent[str].emit("max")

    def on_zoom(self, direction, pos):
        """Pass the zoom signal up to the AudioView

        Trieds to keep the cursor position fixed.

        direction: +1, -1 direction of zoom
        pos: (t, _) of mouse cursor during zoom.
        """
        t_start, t_stop = self.view_state.get("current_range")
        t_cursor = t_start + pos[0] * (t_stop - t_start)

        if direction > 0:
            self.events.zoomEvent[int, float].emit(1, t_cursor)
        elif direction < 0:
            self.events.zoomEvent[int, float].emit(-1, t_cursor)

    def on_drag_in_progress(self, start, end):
        """Draws vertical lines deliminting the edges of the drag

        If in spectrogram only mode, also draws horizontal lines
        If in amp env mode, draws the threshold line
        """
        dx = end.x() - start.x()
        self._clear_drag_lines()

        self.spectrogram_plot.selected_range_line_start.setValue(start.x())
        self.spectrogram_plot.selected_range_line_stop.setValue(end.x())
        self.spectrogram_plot.selected_range_line_start.setVisible(True)
        self.spectrogram_plot.selected_range_line_stop.setVisible(True)

        if not self.view_state.has("select_by_threshold"):
            self.spectrogram_plot.spec_range_line_start.setValue(start.y())
            self.spectrogram_plot.spec_range_line_stop.setValue(end.y())
            self.spectrogram_plot.spec_range_line_start.setVisible(True)
            self.spectrogram_plot.spec_range_line_stop.setVisible(True)

        else:
            curve_1 = self._draw_drag_curves(self.spectrogram_plot, start, end)
            self.drag_curves = {
                self.spectrogram_plot: curve_1,
            }

    def on_drag_complete(self, start, end):
        """Manage the end of a drag event

        A drag event can come in multiple flavors. The main ones are

        (1) Dragging to denote a time range
        (2) Dragging to draw a threshold line
        (3) Dragging to box a range of frequencies in time

        All three care about the time of the drag start and drag stop. (2) and (3)
        care about the y values in the amp env and spectrogram scales respectively.

        (2) Cares about the ampenv values at drag start and stop to threshold
            sections of the ampenv above that line
        (3) Cares about frequency values (of spectrogram space) of drag start
            and drag stop (so that detection can be done after bandpassing
            in that range, )
        """
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        start_dt = start.x() / spec_sample_rate
        end_dt = end.x() / spec_sample_rate
        base_t = self.view_state.get("current_range")[0]
        start_t = base_t + start_dt
        end_t = base_t + end_dt

        if self.view_state.has("select_by_threshold") and self._current_ampenv_yscale:
            self.view_state.set(
                "selected_threshold_line",
                (
                    start.y() * self._current_ampenv_yscale,
                    end.y() * self._current_ampenv_yscale
                )
            )

        spec_range = sorted(np.array([
            start.y() * self._current_spectrogram_yscale,
            end.y() * self._current_spectrogram_yscale
        ]))
        self.view_state.set(
            "selected_spec_range",
            tuple(spec_range)
        )
        self.view_state.set(
            "source_focus",
            self.source_idx
        )

        self.events.rangeSelected.emit(start_t, end_t)

        if self.state.has("autosearch"):
            self.on_find_in_selection()

    def on_click(self, loc, event=None):
        """Clear the selection when a point is clicked (and not dragged)
        """
        # See if the click was within a known interval
        # If it was, bring up a context menu for that interval. Otherwise,
        # simply clear the currently selected range
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        base_t = self.view_state.get("current_range")[0]
        t = base_t + loc.x() / spec_sample_rate

        if isinstance(self.source.get("intervals"), pd.DataFrame):
            df = self.source.get("intervals")
            selector = (df["t_start"] <= t) & (df["t_stop"] >= t)
            match = df[selector]

            # TODO (kevin): perhaps we should highlight this section of the spectrogram somehow?
            if len(match) and event is not None:
                pos = event.screenPos()

                ## Example of creating a menu - probably should move this to a new function
                self.menu = gui.QMenu()
                search = gui.QAction("DC", self.menu)
                delete = gui.QAction("Te", self.menu)
                merge = gui.QAction("Song", self.menu)

                # TODO (kevin): these should trigger the labeling of the selected interval
                # in the dataframe
                self.menu.addAction(search)
                self.menu.addAction(merge)
                self.menu.addAction(delete)
                self.menu.popup(QtCore.QPoint(pos.x(), pos.y()))
                # return

        self.events.rangeSelected.emit(None, None)

    def on_range_selected(self, start_t, end_t):
        """Handle when a range is selected or deselected
        """
        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        if start_t is None or end_t is None:
            self._clear_drag_lines()
            if self.view_state.has("selected_range"):
                self.view_state.clear("selected_range")
            self.spectrogram_plot.selected_range_line_start.setVisible(False)
            self.spectrogram_plot.selected_range_line_stop.setVisible(False)
            self.spectrogram_plot.spec_range_line_start.setVisible(False)
            self.spectrogram_plot.spec_range_line_stop.setVisible(False)
        else:
            start_t, end_t = min(start_t, end_t), max(start_t, end_t)
            self.view_state.set("selected_range", (start_t, end_t))

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

    def on_find_in_selection(self, modify=0, mode="broadband"):
        """Searches for calls in selected window

        modify:
            +1: increase the number of calls detected in the window
            by decreasing the threshold, ignore width, and fuse duration
            -1: decrease "
            0: just do a default search

        (1) Use the ampenv threshold line first. if that doesn't make sense
        (2) use the spectrogram range function. then fall back.
        """
        if not self.view_state.has("selected_range"):
            return

        if self.source.get("readonly"):
            return

        t0, t1 = self.view_state.get("selected_range")

        if isinstance(self.source.get("intervals"), pd.DataFrame):
            df = self.source.get("intervals")
            selector = (
                (df["t_start"] >= t0) & (df["t_stop"] <= t1) |
                (df["t_start"] < t0) & (df["t_stop"] > t0) |
                (df["t_start"] < t1) & (df["t_stop"] > t1)
            )
            old_count = len(df[selector])
        else:
            old_count = 0

        if self.view_state.has("selected_threshold_line"):
            # (1) Search by amplitude threshold
            y0, y1 = self.view_state.get("selected_threshold_line")
            # Map the xdata onto time values
            m = (y1 - y0) / (t1 - t0)
            b = y0 - t0 * m
            def _thresh(t):
                return m * t + b
            xdata, ydata = self.ampenv_plot.getData()
            tdata = self.view_state.get("current_range")[0] + xdata / read_default.SPEC_SAMPLE_RATE
            range_selector = (tdata >= t0) & (tdata < t1)
            ythresh = _thresh(tdata[range_selector])
            yvalues = self._current_ampenv_yscale * ydata[range_selector]

            events = threshold_events(
                yvalues,
                ythresh,
                polarity=1,
                sampling_rate=1,
                ignore_width=2,
                min_size=2,
                fuse_duration=2
            )
            events = [
                [tdata[range_selector][e[0]], tdata[range_selector][e[1] - 1]]
                for e in events
            ]
        elif modify == 0 or old_count == 0:
            # (2) Search by frequency band selected
            if self.view_state.has("selected_spec_range"):
                y0, y1 = self.view_state.get("selected_spec_range")
            else:
                y0, y1 = 1000.0, 8000.0
            events = threshold_all_events(
                self.state.get("sound_object"),
                window_size=None,
                channel=self.source["channel"],
                t_start=t0,
                t_stop=t1,
                ignore_width=0.005,
                min_size=0.005,
                fuse_duration=0.01,
                threshold_z=2.0,
                highpass=y0,
                lowpass=y1,
                amp_env_mode=mode
            )
        elif modify in (-1, 1):
            # if self.view_state.has("selected_spec_range"):
            #     y0, y1 = self.view_state.get("selected_spec_range")
            # else:
            y0, y1 = 1000.0, 8000.0
            # use a lower / higher threshold
            t_arr, sig = self.state.get("sound_object").time_slice(t0, t1)
            t_arr += t0
            sig -= np.mean(sig, axis=0)

            amp_env = get_amplitude_envelope(
                sig[:, self.source["channel"]],
                fs=self.state.get("sound_object").sampling_rate,
                highpass=y0,
                lowpass=y1,
                rectify_lowpass=200.0,
                mode="broadband"
            )

            threshold_adjuster = ThresholdAdjuster(t_arr, amp_env, df[selector])
            low, mid, high = threshold_adjuster.estimate()
            threshold = high if modify == 1 else low

            events = threshold_events(
                amp_env,
                threshold,
                polarity=1,
                sampling_rate=self.state.get("sound_object").sampling_rate,
                ignore_width=0.005,
                min_size=0.005,
                fuse_duration=0.01,
            )
            events = [
                [t_arr[e[0]], t_arr[e[1] - 1]]
                for e in events
            ]

        # Replaces the relevant rows in the dataframe
        new_rows = [{
            "t_start": event[0],
            "t_stop": event[1],
        } for event in events]

        if not len(new_rows):
            self._draw_intervals()
            return

        if isinstance(self.source.get("intervals"), pd.DataFrame):
            new_df = df[~selector].copy()
            new_df = new_df.append(new_rows, ignore_index=False, sort=True)
        else:
            new_df = pd.DataFrame(new_rows)

        new_df = new_df.sort_values(by="t_start")

        self.source["intervals"] = new_df
        self._draw_intervals()

    def on_delete_selection(self):
        """Delete from [intervals] any intervals that are partial contained within
        the currently selected range.
        """
        if not self.view_state.has("selected_range"):
            return

        if self.source.get("readonly"):
            return

        if "intervals" not in self.source:
            return

        selection_start, selection_stop = self.view_state.get("selected_range")
        df = self.source["intervals"]
        selector = (
            ((df["t_start"] >= selection_start) & (df["t_stop"] <= selection_stop)) |
            ((df["t_start"] < selection_stop) & (df["t_stop"] > selection_stop)) |
            ((df["t_start"] < selection_start) & (df["t_stop"] > selection_start))
        )
        self.source["intervals"] = df[~selector].copy()
        self._draw_intervals()

    def on_merge_selection(self):
        if not self.view_state.has("selected_range"):
            return

        if self.source.get("readonly"):
            return

        if "intervals" not in self.source:
            return

        selection_start, selection_stop = self.view_state.get("selected_range")
        df = self.source["intervals"]
        selector = (
            ((df["t_start"] >= selection_start) & (df["t_stop"] <= selection_stop)) |
            ((df["t_start"] < selection_stop) & (df["t_stop"] > selection_stop)) |
            ((df["t_start"] < selection_start) & (df["t_stop"] > selection_start))
        )

        if not len(df[selector]):
            # Instead, select all and then merge
            self.on_find_in_selection(0)
            return

        # Merge those selected into a single row
        # First: make a new df with the non-selected rows
        new_df = df[~selector].copy()

        # Create a single row encapsulating the selected intervals
        new_row = {
            "t_start": np.min(df[selector]["t_start"]),
            "t_stop": np.max(df[selector]["t_stop"]),
        }

        # Add the row to the dataframe and resort by t_start
        new_df = new_df.append([new_row], ignore_index=False, sort=True)
        new_df = new_df.sort_values(by="t_start")

        self.source["intervals"] = new_df.copy()
        self._draw_intervals()

    def _update_highlighted_range(self):
        base_t, base_end_t = self.view_state.get("current_range")
        win_size = read_default.WINDOW_SIZE

        spec_sample_rate = read_default.SPEC_SAMPLE_RATE

        if self._highlight_rect is not None:
            self.spectrogram_plot.removeItem(self._highlight_rect)
            self._highlight_rect = None

        if self.view_state.has("highlighted_range"):
            start_t, end_t = self.view_state.get("highlighted_range")

            start_frac = (start_t - base_t) / win_size
            duration_frac = (end_t - start_t) / win_size

            viewbox = self.spectrogram_plot.getPlotItem().getViewBox()
            ((xmin, xmax), (ymin, ymax)) = viewbox.viewRange()
            self._highlight_rect = gui.QGraphicsRectItem(
                xmin + (start_frac * (xmax - xmin)),
                ymax - (ymax / 80),
                duration_frac * (xmax - xmin),
                ymax / 40
            )
            self._highlight_rect.setPen(pg.mkPen(None))
            self._highlight_rect.setBrush(pg.mkBrush("g"))
            self.spectrogram_plot.addItem(self._highlight_rect)

    def _draw_intervals(self):
        """Draw rectangles labeling events for the chosen source"""
        viewbox = self.spectrogram_plot.getPlotItem().getViewBox()

        for rect in self._drawn_intervals:
            viewbox.removeItem(rect)
        self._drawn_intervals = []

        intervals = self.source.get("intervals", [])
        if not len(intervals):
            return

        t1, t2 = self.view_state.get("current_range")
        start_index = np.searchsorted(intervals["t_start"], t1)
        if start_index > 0:
            start_index -= 1

        win_size = read_default.WINDOW_SIZE
        for idx in range(start_index, len(intervals)):
            start_t, end_t = intervals.iloc[idx][["t_start", "t_stop"]]
            if start_t > t2:
                break

            interval_contained = (
                (t1 <= start_t <= t2) or
                (t1 <= end_t <= t2) or
                (start_t <= t1 <= t2 <= end_t)
            )
            if interval_contained:
                start_frac = (start_t - t1) / win_size
                duration_frac = (end_t - start_t) / win_size

                ((xmin, xmax), (_, ymax)) = viewbox.viewRange()

                rect = gui.QGraphicsRectItem(
                    xmin + (start_frac * (xmax - xmin)),
                    ymax - (3 * ymax / 40),
                    duration_frac * (xmax - xmin),
                    ymax / 20
                )
                rect.setPen(pg.mkPen(None))
                # TODO (kevin): choose color based on label
                color = "#8fcfd1" if not self.source.get("readonly") else "#df5e88"
                rect.setBrush(pg.mkBrush(color))
                viewbox.addItem(rect)
                self._drawn_intervals.append(rect)


class VocalPeriodsView(widgets.QScrollArea):
    """Show and jump to windows to highlight them"""

    def __init__(self, parent=None, events=None):
        super().__init__(parent)
        self.state = AppState()
        self.view_state = ViewState()
        self.events = events
        self.thread = None

        self.currently_selected = None

        self.init_ui()

        self.events.dataLoaded.connect(self.on_data_loaded)

    def init_ui(self):
        self.mainLayout = widgets.QHBoxLayout()

        self.vocalPeriodsLayout = widgets.QHBoxLayout()

        self.detectButton = widgets.QPushButton("Detect Potential\nVocal Periods", self)
        self.detectButton.setGeometry(200, 150, 100, 100)
        self.detectButton.setStyleSheet("width: 150px; height: 150px;")

        self.mainLayout.addWidget(self.detectButton)
        self.mainLayout.addLayout(self.vocalPeriodsLayout)
        self.mainLayout.addStretch()

        self.setLayout(self.mainLayout)

        self.all_vocal_period_buttons = []

        self.detectButton.clicked.connect(self.on_detect)

    def clear_vocal_periods(self):
        for i in reversed(range(self.vocalPeriodsLayout.count())):
            item = self.vocalPeriodsLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.all_vocal_period_buttons = []

    def _reset_thread(self):
        if self.thread is not None:
            self.thread.exit()
        self.thread = QThread(self)
        return self.thread

    def closeEvent(self):
        if self.thread is not None:
            self.thread.stop()

    def on_data_loaded(self):
        self.clear_vocal_periods()

    def on_detect(self):
        """
        Any custom detection hooks would go here
        """
        self.clear_vocal_periods()
        self._last_spec_key = uuid.uuid4().hex
        if self.state.has("autodetected_periods"):
            df = self.state.get("autodetected_periods")

            # Convert the detected periods (which may represent individual calls)
            # into larger chunks (split up if there are more than 20s of silence)
            def _chunk(df, silence=20):
                chunks = []
                current_chunk = {}
                for i in range(len(df)):
                    row = df.iloc[i]
                    if not current_chunk:
                        current_chunk["t_start"] = row["t_start"]
                        current_chunk["t_stop"] = row["t_stop"]
                    else:
                        if row["t_start"] - current_chunk["t_stop"] > silence:
                            chunks.append(current_chunk)
                            current_chunk = {
                                "t_start": row["t_start"],
                                "t_stop": row["t_stop"]
                                }
                        else:
                            current_chunk["t_stop"] = row["t_stop"]
                return pd.DataFrame(chunks)

            df = _chunk(df)

            self.on_vocal_periods_complete(
                self._last_spec_key,
                [
                    (df.iloc[i]["t_start"], df.iloc[i]["t_stop"])
                    for i in range(len(df))
                ]
            )
            return
        else:
            return

        # Detect on all channels with sources - merge them, then present them as options
        audio_signal = self.state.get("sound_object")
        channels = np.unique([source["channel"] for source in self.state.get("sources")])

        self.worker = VocalPeriodsWorker(
            self._last_spec_key,
            channels,
            audio_signal,
        )
        self.worker.finished.connect(partial(
            self.on_vocal_periods_complete,
        ))
        self._reset_thread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.compute)
        self.thread.start()

    def on_vocal_periods_complete(self, key, vocal_periods):
        if key != self._last_spec_key:
            return

        for i, period in enumerate(vocal_periods):
            button = widgets.QPushButton("{:.2f}\nto\n{:.2f}".format(period[0], period[1]))
            button.setCheckable(True)
            button.clicked.connect(partial(self.on_vocal_period_clicked, i, period))
            self.vocalPeriodsLayout.addWidget(button)
            button.setStyleSheet("width:100px; height: 100px;")
            self.all_vocal_period_buttons.append(button)

    def on_vocal_period_clicked(self, idx, period):
        # check the relevant button
        self.currently_selected = idx

        for i, button in enumerate(self.all_vocal_period_buttons):
            button.setChecked(i == idx)

        t1, t2 = period
        t_center = np.mean([t1, t2])
        self.events.setPosition[object, object].emit(t1, TimeScrollManager.ALIGN_LEFT)
        self.view_state.set("highlighted_range", (t1, t2))
        self.events.rangeHighlighted.emit(t1, t2)
