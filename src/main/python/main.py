import sys
sys.path.append("code/soundsep")

import glob
import os
import re
from functools import partial

import hdbscan
import numpy as np
import pyqtgraph as pg
import scipy
import sounddevice as sd
import umap
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtCore import (Qt, QObject, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets
from sklearn.decomposition import PCA

from soundsig.sound import spectrogram
from soundsig.signal import bandpass_filter

from audio_utils import get_amplitude_envelope
from interfaces.audio import LazyMultiWavInterface, LazyWavInterface


MAX_RECENT_FILES = 5


def _spec2icon(spec, dBNoise=40):
    """Convert a spectrogram into a grayscale Qt icon

    Chooses the channel with the higher amplitude and converts to dB
    """
    spec = np.abs(spec)
    if spec.ndim == 3:
        best_ch = np.where(spec == np.max(spec))[0][0]
        spec = spec[best_ch]

    # spec must be flipped upside down for icon
    spec = 20 * np.log10(spec)
    min_val = np.max(spec) - dBNoise
    spec[spec < min_val] = min_val
    spec = spec - np.min(spec)
    spec = spec[::-1]

    # Convert to grayscale pixel values
    spec = 255 * spec / np.max(spec)
    spec = np.require(spec, np.uint8, "C")

    qtimage = gui.QImage(
        spec.data,
        spec.shape[1],
        spec.shape[0],
        gui.QImage.Format_Indexed8
    )
    icon = gui.QPixmap.fromImage(qtimage)
    return gui.QIcon(icon)


class Singleton():
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class AppData(Singleton):
    def __init__(self):
        Singleton.__init__(self)
        self.reset()

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def update(self, update_dict):
        for key, val in update_dict.items():
            self._data[key] = val

    def has(self, key):
        return key in self._data

    def clear(self, key):
        del self._data[key]

    def reset(self):
        self._data = {}


class BackgroundEmbedding(QObject):
    """Async worker for computing umap embedding"""
    finished = pyqtSignal(object)

    def __init__(self, data):
        super().__init__()
        self.data = data

    @pyqtSlot()
    def compute(self):
        points = self.data.reshape(len(self.data), -1)
        points = PCA(n_components=20, whiten=True).fit_transform(points)
        embedding = umap.UMAP(
            n_neighbors=10, repulsion_strength=100.0, min_dist=0.9
        ).fit_transform(points)
        self.finished.emit(embedding)


class App(widgets.QMainWindow):
    """Main App instance with logic for file io
    """
    # Dictionary of data with keys wav, intervals, spectrograms, labels
    signalLoadedData = pyqtSignal(object)

    # Cluster label, array of spectrograms, array of indices into original data
    clusterSelected = pyqtSignal(int, object, object)

    # Overall index of snippet selected
    snippetSelected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.settings = QSettings("Theuniseen Lab", "Sound Separation")
        self.loaded_data = AppData()

        self.init_ui()
        self.init_actions()
        self.init_menus()
        self.update_open_recent_actions()

        self.thread = None

        if self.settings.value("OPEN_RECENT", []):
            self.load_dir(self.settings.value("OPEN_RECENT")[-1])

    def init_actions(self):
        self.open_directory_action = widgets.QAction("Open Directory", self)
        self.open_directory_action.triggered.connect(self.run_directory_loader)

        self.open_recent_actions = []
        for i in range(MAX_RECENT_FILES):
            action = widgets.QAction("", self)
            action.setVisible(False)
            action.triggered.connect(self.load_dir)
            self.open_recent_actions.append(action)

        self.run_embedding_action = widgets.QAction("Run UMAP", self)
        self.run_embedding_action.triggered.connect(self.run_embedding)
        self.run_labeler_action = widgets.QAction("Run HDBSCAN Labeling", self)
        self.run_labeler_action.triggered.connect(self.run_labeler)
        self.save_embedding_action = widgets.QAction("Save Embedding", self)
        self.save_embedding_action.triggered.connect(self.save_embedding)
        self.save_labels_action = widgets.QAction("Save Labels", self)
        self.save_labels_action.triggered.connect(self.save_labels)

    def update_open_recent_actions(self):
        recently_opened = self.settings.value("OPEN_RECENT", [])
        for i in range(MAX_RECENT_FILES):
            if i < len(recently_opened):
                self.open_recent_actions[i].setText(recently_opened[-i])
                self.open_recent_actions[i].setData(recently_opened[-i])
                self.open_recent_actions[i].setVisible(True)
            else:
                self.open_recent_actions[i].setText(None)
                self.open_recent_actions[i].setData(None)
                self.open_recent_actions[i].setVisible(False)
        if not len(recently_opened):
            self.openRecentMenu.setDisabled(True)
        else:
            self.openRecentMenu.setDisabled(False)

    def init_ui(self):
        self.setWindowTitle(self.title)

    def init_menus(self):
        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(self.open_directory_action)
        self.openRecentMenu = fileMenu.addMenu("&Open Recent")
        for i in range(MAX_RECENT_FILES):
            self.openRecentMenu.addAction(self.open_recent_actions[i])
        fileMenu.addSeparator()
        fileMenu.addAction(self.save_embedding_action)
        fileMenu.addAction(self.save_labels_action)

        analysisMenu = mainMenu.addMenu("&Analysis")
        analysisMenu.addAction(self.run_embedding_action)
        analysisMenu.addAction(self.run_labeler_action)

        self.display_viewer()

    def display_viewer(self):
        self.main_view = MainView(self)
        self.setCentralWidget(self.main_view)
        self.show()

    def closeEvent(self, evt):
        if self.thread:
            self.thread.terminate()

    def run_embedding(self):
        """Compute a pca->umap embedding for the detected audio data async"""
        specs = self.loaded_data.get("spectrograms")
        self.worker = BackgroundEmbedding(specs)
        self.run_embedding_action.setDisabled(True)
        self.run_labeler_action.setDisabled(True)
        self.worker.finished.connect(self._on_embedding_completed)

        if self.settings.value("ASYNC_FLAG", False):
            self._reset_thread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.compute)
            self.thread.start()
        else:
            self.worker.compute()

    def _on_embedding_completed(self, embedding):
        self.run_embedding_action.setDisabled(False)
        self.run_labeler_action.setDisabled(False)

        self.loaded_data.set("embedding", embedding)
        self.signalLoadedData.emit(self.loaded_data)

    def _reset_thread(self):
        self.thread = QThread(self)
        return self.thread

    def run_labeler(self):
        """Label the loaded data using a pca->umap embedding and hdbscan
        """
        if self.loaded_data.get("labels") is not None:
            msg = "Are you sure you want to overwrite current labels?"
            reply = widgets.QMessageBox.question(
                    self,
                    'Confirmation',
                    msg,
                    widgets.QMessageBox.Yes,
                    widgets.QMessageBox.No)
            if reply == widgets.QMessageBox.No:
                return

        if not self.loaded_data.has("embedding"):
            self.run_embedding()
            if self.settings.value("ASYNC_FLAG", False):
                self.worker.finished.connect(self._run_labeler)
            else:
                self._run_labeler()
        else:
            self._run_labeler()

    def _run_labeler(self):
        embedding = self.loaded_data.get("embedding")
        labels = hdbscan.HDBSCAN().fit_predict(embedding)
        self.loaded_data.set("labels", labels)
        self.signalLoadedData.emit(self.loaded_data)

    def run_directory_loader(self):
        """Dialog to read in a directory of wav files and intervals

        At some point we may have the gui generate the intervals file if it doesn't
        exist yet, but for now it must be precomputed.

        The directory should contain the following files (one wav per channel):
            - ch0.wav
            - ch1.wav
            - ch2.wav
            ...
            - intervals.npy
            - spectrograms.npy
        """
        options = widgets.QFileDialog.Options()
        selected_file = widgets.QFileDialog.getExistingDirectory(
            self,
            "Load directory",
            self.settings.value("OPEN_RECENT", ["."])[-1],
            options=options
        )

        if selected_file:
            self.settings.setValue("file/wavfile", selected_file)
            self.load_dir(selected_file)

    def save_embedding(self):
        if self.loaded_data.has("embedding"):
            np.save(self.embedding_file, self.loaded_data.get("embedding"))

    def save_labels(self):
        if self.loaded_data.has("labels"):
            np.save(self.labels_file, self.loaded_data.get("labels"))

    def load_dir(self, selected_file):
        if not os.path.isdir(selected_file):
            raise IOError("{} is not a directory".format(selected_file))

        # Update the open recent menu item
        open_recent = self.settings.value("OPEN_RECENT", [])
        try:
            idx = open_recent.index(selected_file)
        except ValueError:
            open_recent.append(selected_file)
        else:
            open_recent.pop(idx)
            open_recent.append(selected_file)
        open_recent = open_recent[-MAX_RECENT_FILES:]
        self.settings.setValue("OPEN_RECENT", open_recent)
        self.update_open_recent_actions()

        self.loaded_data.reset()

        self.data_directory = os.path.join(selected_file, "outputs")
        wav_files = glob.glob(os.path.join(selected_file, "ch[0-9]*.wav"))
        self.intervals_file = os.path.join(self.data_directory, "intervals.npy")
        self.spectrograms_file = os.path.join(self.data_directory, "spectrograms.npy")
        self.labels_file = os.path.join(self.data_directory, "labels.npy")
        self.embedding_file = os.path.join(self.data_directory, "embedding.npy")

        if not len(wav_files):
            raise IOError("No files matching {} found".format(
                    os.path.join(selected_file, regexp)))
        if not os.path.exists(self.intervals_file):
            raise IOError("No file named {} exists".format(self.intervals_file))
        if not os.path.exists(self.spectrograms_file):
            raise IOError("No file named {} exists".format(self.spectrograms_file))

        # TODO (kevin): Make it optional for intervals, spectrograms, and labels
        # to exist... we should be able to generate these
        if len(wav_files) > 1:
            wav_object = LazyMultiWavInterface.create_from_directory(selected_file)
        else:
            wav_object = LazyWavInterface(wav_files[0])

        self.loaded_data.set("wav", wav_object)
        self.loaded_data.set("intervals", np.load(self.intervals_file)[()])
        self.loaded_data.set("spectrograms", np.load(self.spectrograms_file)[()])
        if os.path.exists(self.embedding_file):
            self.loaded_data.set("embedding", np.load(self.embedding_file)[()])
        if os.path.exists(self.labels_file):
            self.loaded_data.set("labels", np.load(self.labels_file)[()])
        self.loaded_data.set("loaded_dir", selected_file)
        self.signalLoadedData.emit(self.loaded_data)


class MainView(widgets.QWidget):
    """Container for the overall layout of the app

    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_plots()

        self.loaded_data = AppData()

        self.parent().signalLoadedData.connect(self.on_data_load)

    def init_ui(self):
        self.topBar = widgets.QHBoxLayout()
        self.topBarLabel = widgets.QLabel("")
        self.topBarLabel.setDisabled(True)
        self.topBar.addWidget(self.topBarLabel)
        self.topBar.addStretch(1)
        self.topLeftBox = widgets.QGroupBox("Scatter 2D")
        self.topRightBox = widgets.QGroupBox("Sound Viewer")
        self.bottomLeftBox = widgets.QGroupBox("Clusters")
        self.bottomRightBox = widgets.QGroupBox("Detected Snippets")
        self.footerBox = widgets.QGroupBox("")

        self.bottomSplitter = widgets.QSplitter(Qt.Horizontal)
        self.bottomSplitter.addWidget(self.bottomLeftBox)
        self.bottomSplitter.addWidget(self.bottomRightBox)

        self.topSplitter = widgets.QSplitter(Qt.Horizontal)
        self.topSplitter.addWidget(self.topLeftBox)
        self.topSplitter.addWidget(self.topRightBox)

        self.mainLayout = widgets.QGridLayout()
        self.mainLayout.addLayout(self.topBar, 0, 0)
        self.mainLayout.addWidget(self.topSplitter, 1, 0, 1, 6)
        self.mainLayout.addWidget(self.bottomSplitter, 2, 0, 1, 6)

        self.mainLayout.setRowStretch(1, 1)
        self.mainLayout.setRowStretch(2, 1)

        for col in range(6):
            self.mainLayout.setColumnStretch(col, 1)

        self.setLayout(self.mainLayout)

        self.topLeftBox.setDisabled(True)
        self.topRightBox.setDisabled(True)

    def init_plots(self):
        self.scatter_widget = Scatter2DView(
            None,
            data_loaded_signal=self.parent().signalLoadedData
        )

        self.spectrogram_widget = AudioView(
            None,
            data_loaded_signal=self.parent().signalLoadedData,
            snippet_selected_signal=self.parent().snippetSelected
        )

        self.cluster_select_widget = ClusterSelectView(
            None,
            data_loaded_signal=self.parent().signalLoadedData,
            cluster_selected_signal=self.parent().clusterSelected
        )

        self.snippet_select_widget = SnippetSelectView(
            None,
            cluster_selected_signal=self.parent().clusterSelected,
            snippet_selected_signal=self.parent().snippetSelected
        )

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.scatter_widget)
        layout.addStretch(1)
        self.topLeftBox.setLayout(layout)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_widget)
        layout.addStretch(1)
        self.topRightBox.setLayout(layout)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.cluster_select_widget)
        self.bottomLeftBox.setLayout(layout)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.snippet_select_widget)
        self.bottomRightBox.setLayout(layout)

    def on_data_load(self, data):
        self.topLeftBox.setDisabled(False)
        self.topRightBox.setDisabled(False)
        self.topBarLabel.setText(self.loaded_data.get("loaded_dir"))


class Scatter2DView(widgets.QWidget):
    """Panel for 2d scatter plot of data
    """
    valid_projections = ["pca", "umap"] #, "umap", "tsne"]

    def __init__(self, parent=None, data_loaded_signal=None):
        super().__init__(parent)
        self.init_ui()
        self.loaded_data = AppData()
        data_loaded_signal.connect(self.on_data_load)

    def init_ui(self):
        self.plot = pg.PlotWidget()
        pen = pg.mkPen((200, 200, 250, 127))
        self.scatter = pg.ScatterPlotItem(pen=pen, symbol="o", size=1)
        self.plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.plot.addItem(self.scatter)
        self.plot.plotItem.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addStretch(1)
        self.setLayout(layout)

    def on_data_load(self, data):
        if not self.loaded_data.has("embedding"):
            self.setDisabled(True)
            self.set_data(None)
        else:
            self._embedding = self.loaded_data.get("embedding")
            self.set_data(self._embedding)

    def set_data(self, embedding):
        if embedding is None:
            self.scatter.setData([])
            return

        self.scatter.setData([
            {"pos": [x, y]} for x, y in embedding
        ])

        dyn_range_x = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
        dyn_range_y = np.max(embedding[:, 1]) - np.min(embedding[:, 1])
        self.plot.setLimits(
            xMin=np.min(embedding[:, 0]) - 0.1 * dyn_range_x,
            xMax=np.max(embedding[:, 0]) + 0.1 * dyn_range_x,
            yMin=np.min(embedding[:, 1]) - 0.1 * dyn_range_y,
            yMax=np.max(embedding[:, 1]) + 0.1 * dyn_range_y
        )

        return embedding


class AudioView(widgets.QWidget):
    """Panel for viewing spectrogram of a time period

    1. Channel selection
    2. Play sample audio of selected window
    3. Show spectrogram
    4. Select time range with scrollbar

    Also include a second tab to switch between spectrogram and amplitude
    """
    win_size = 3.0  # seconds
    spec_sample_rate = 500
    spec_freq_spacing = 50
    min_freq = 250
    max_freq = 8000

    def __init__(self, parent=None, data_loaded_signal=None,
            snippet_selected_signal=None):
        super().__init__(parent)
        self.init_ui()
        self.loaded_data = AppData()

        self.view_ch = 0
        self.current_step = 0

        # Set up playback line
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_playback_line)
        self.reset_playback_line()

        data_loaded_signal.connect(self.on_data_load)
        snippet_selected_signal.connect(self.on_snippet_selected)

    def _set_channel(self, ch):
        self.view_ch = ch
        self.update_image()

    def _set_n_channels(self, n):
        """Update radio buttons to match number of channels in current audio file
        """
        for i in reversed(range(self.channelSelectLayout.count())):
            item = self.channelSelectLayout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            else:
                # Remove the spacer
                self.channelSelectLayout.removeItem(item)

        for i in range(n):
            button = widgets.QRadioButton("Ch{}".format(i))
            button.released.connect(partial(self._set_channel, i))
            self.channelSelectLayout.addWidget(button)
            if i == 0:
                # Have channel 0 checked by default
                self.view_ch = 0
                button.setChecked(True)

        # Adds a spacer to keep all items left aligned
        self.channelSelectLayout.addStretch()

        play_button = widgets.QPushButton("Play")
        play_button.released.connect(self.play_audio)
        self.channelSelectLayout.addWidget(play_button)

    def play_audio(self):
        """Play the sound thats in the currently selected window"""
        self.reset_playback_line()
        sd.play(self._sig[:, self.view_ch], self._loaded_wav.sampling_rate, blocking=False)
        self.timer.start(10000 / self.spec_sample_rate)

    def reset_playback_line(self):
        self.timer.stop()
        self._playback_line_pos = -1
        self.spectrogram_plot.playback_line.setValue(self._playback_line_pos)
        self.amplitude_plot.playback_line.setValue(self._playback_line_pos)

    def advance_playback_line(self):
        """Step the playback line forward in time

        Reset the line when it reaches the end
        """
        max_playback_line_pos = int(self.win_size * self.spec_sample_rate)
        if self._playback_line_pos < max_playback_line_pos:
            self._playback_line_pos += 10
            self.spectrogram_plot.playback_line.setValue(self._playback_line_pos)
            self.amplitude_plot.playback_line.setValue(self._playback_line_pos)
        else:
            self.reset_playback_line()

    def init_ui(self):
        ### Spectrogram Plot
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.playback_line = pg.InfiniteLine()
        self.spectrogram_plot.snippet_line_start = pg.InfiniteLine()
        self.spectrogram_plot.snippet_line_stop = pg.InfiniteLine()
        self.spectrogram_plot.addItem(self.image)
        self.spectrogram_plot.addItem(self.spectrogram_plot.playback_line)
        self.spectrogram_plot.addItem(self.spectrogram_plot.snippet_line_start)
        self.spectrogram_plot.addItem(self.spectrogram_plot.snippet_line_stop)
        self.spectrogram_plot.hideAxis("left")
        self.spectrogram_plot.hideAxis("bottom")
        self.spectrogram_plot.setLimits(
            xMin=0,
            xMax=int(self.win_size * self.spec_sample_rate),
            yMin=0,
            yMax=int((self.max_freq - self.min_freq) / self.spec_freq_spacing)
        )

        ### Amplitude Plot
        # The amplitude plot will use the same sampling rate as the spectrograms
        # ideally to make the playback line simpler...
        self.amplitude_plot = pg.PlotWidget()
        self.amplitude_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.amplitude_plot.playback_line = pg.InfiniteLine()
        self.amplitude_plot.snippet_line_start = pg.InfiniteLine()
        self.amplitude_plot.snippet_line_stop = pg.InfiniteLine()
        self.amplitude_plot.addItem(self.amplitude_plot.playback_line)
        self.amplitude_plot.addItem(self.amplitude_plot.snippet_line_start)
        self.amplitude_plot.addItem(self.amplitude_plot.snippet_line_stop)
        self.amplitude_plot.hideAxis("left")
        self.amplitude_plot.hideAxis("bottom")
        self.amplitude_plot.setLimits(
            xMin=0,
            xMax=int(self.win_size * self.spec_sample_rate),
            yMin=0,
            yMax=1
        )

        # Put the plots in a tab widget
        self.tab_panel = widgets.QTabWidget(self)
        self.tab_panel.addTab(self.spectrogram_plot, "Spectrogram")
        self.tab_panel.addTab(self.amplitude_plot, "Amplitude Envelope")

        # Other widgets in this panel
        self.channelSelectLayout = widgets.QHBoxLayout()
        self._set_n_channels(1)

        self.scrollbar = widgets.QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(100)
        self.scrollbar.valueChanged.connect(self.change_range)

        self.windowInfoLayout = widgets.QHBoxLayout()
        self.time_label = widgets.QLabel()
        self.windowInfoLayout.addWidget(self.time_label)
        self.windowInfoLayout.addStretch()

        layout = widgets.QVBoxLayout()
        layout.addLayout(self.channelSelectLayout)
        layout.addWidget(self.tab_panel)
        layout.addWidget(self.scrollbar)
        layout.addLayout(self.windowInfoLayout)
        layout.addStretch(1)
        self.setLayout(layout)

    def on_data_load(self, data):
        self._loaded_wav = self.loaded_data.get("wav")
        self._loaded_intervals = self.loaded_data.get("intervals")
        self.update_scroll_bar()
        self.change_range(0)
        self._set_n_channels(self._loaded_wav.n_channels)

    def on_snippet_selected(self, idx):
        t0, t1 = self._loaded_intervals[idx]
        midpoint = np.mean([t0, t1])
        approx_start_time = midpoint - (self.win_size / 2)
        approx_step = approx_start_time // self.t_step
        self.change_range(approx_step)

        line_pos_start = (t0 - approx_step * self.t_step) * self.spec_sample_rate
        line_pos_stop = (t1 - approx_step * self.t_step) * self.spec_sample_rate

        self.spectrogram_plot.snippet_line_start.setValue(line_pos_start)
        self.spectrogram_plot.snippet_line_stop.setValue(line_pos_stop)
        self.amplitude_plot.snippet_line_start.setValue(line_pos_start)
        self.amplitude_plot.snippet_line_stop.setValue(line_pos_stop)

    def update_scroll_bar(self):
        """Set scrollbar params to match current audio file length and window size"""
        # Set the scrollbar values
        t_last = self._loaded_wav.t_max - self.win_size
        page_step = 10
        single_step = 1
        steps = int(np.ceil(t_last / self.win_size)) * page_step

        self.t_step = self.win_size / 10
        self.scrollbar.setValue(0)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(steps)
        self.scrollbar.setSingleStep(single_step)
        self.scrollbar.setPageStep(page_step)

    def change_range(self, new_value):
        self.spectrogram_plot.snippet_line_start.setValue(-1)
        self.spectrogram_plot.snippet_line_stop.setValue(-1)
        self.amplitude_plot.snippet_line_start.setValue(-1)
        self.amplitude_plot.snippet_line_stop.setValue(-1)
        self.current_step = new_value
        self.scrollbar.setValue(new_value)
        self.update_image()
        self.update_time_label()

    def update_image(self):
        t1 = self.t_step * self.current_step
        t2 = t1 + self.win_size

        t_arr, sig = self._loaded_wav.time_slice(t1, t2)
        sig -= np.mean(sig, axis=0)

        sd.stop()
        self.reset_playback_line()

        self._update_image_spectrogram(sig)
        self._update_image_amplitude(sig)

    def _update_image_spectrogram(self, sig):
        self._sig = sig
        t_spec, f_spec, spec, _ = spectrogram(
            sig[:, self.view_ch],
            self._loaded_wav.sampling_rate,
            spec_sample_rate=self.spec_sample_rate,
            freq_spacing=self.spec_freq_spacing,
            min_freq=self.min_freq,
            max_freq=self.max_freq, cmplx=False
        )

        logspec = 20 * np.log10(np.abs(spec))
        max_b = logspec.max()
        min_b = logspec.max() - 40
        logspec[logspec < min_b] = min_b

        self.image.setImage(logspec.T)

    def _update_image_amplitude(self, sig):
        highlighter_pen = pg.mkPen((29, 224, 32, 255))
        bg_pen = pg.mkPen((204, 204, 204, 127))

        for ch in range(sig.shape[1]):
            amp_env = get_amplitude_envelope(sig[:, ch])
            # Downsample amp_env to spectrogram sampling rate
            downsample_to_n_samples = int(self.win_size * self.spec_sample_rate)
            amp_env = scipy.signal.resample(amp_env, downsample_to_n_samples)
            self.amplitude_plot.plot(
                list(np.arange(len(amp_env))),
                list(amp_env),
                clear=True if ch == 0 else False,
                pen=highlighter_pen if ch == self.view_ch else bg_pen
            )

        # Need to add the overlaied lines again since they were cleared by
        # the plot function above.
        self.amplitude_plot.addItem(self.amplitude_plot.playback_line)
        self.amplitude_plot.addItem(self.amplitude_plot.snippet_line_start)
        self.amplitude_plot.addItem(self.amplitude_plot.snippet_line_stop)

        self.amplitude_plot.setLimits(
            xMin=0,
            xMax=len(amp_env),
            yMin=0,
            yMax=np.max(sig)
        )

    def update_time_label(self):
        t1 = self.t_step * self.current_step
        t2 = t1 + self.win_size
        self.time_label.setText("{:.2f}s - {:.2f}s".format(t1, t2))


class ClusterSelectView(widgets.QScrollArea):
    """A grid of pressable buttons for every cluster currently seen"""

    n_rows = 2

    def __init__(self, parent=None, data_loaded_signal=None, cluster_selected_signal=None):
        super().__init__(parent)
        self.init_ui()
        self.cluster_selected_signal = cluster_selected_signal
        data_loaded_signal.connect(self.on_data_load)

    def on_data_load(self, data):
        self._labels = data.get("labels")
        self._spectrograms = data.get("spectrograms")
        if not data.has("labels"):
            self.setDisabled(True)
        else:
            self.setDisabled(False)
        self.reset_buttons()

    def init_ui(self):
        self.frame = widgets.QGroupBox()
        self.layout = widgets.QGridLayout()
        self.frame.setLayout(self.layout)
        self.setWidget(self.frame)
        self.setWidgetResizable(True)

    def reset_buttons(self):
        self._button_positions = {}
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            else:
                raise Exception("Not supposed to be anything else")

        for idx, l in enumerate(np.unique(self._labels)):
            row = idx % self.n_rows
            col = idx // self.n_rows
            cluster_select = ImgButton(
                self._get_cluster_icon(l),
                l,
                button_callback=self._button_callback,
                radio=True,
                parent=self
            )
            self._button_positions[l] = (row, col)
            self.layout.addWidget(cluster_select, row, col, 1, 1)

    def _button_callback(self, label):
        self._button_positions = {}
        self.cluster_selected_signal.emit(
            label,
            self._spectrograms[self._labels == label],
            np.where(self._labels == label)[0]
        )

        # Manual implementation of mutually exclusive radio buttons
        # - only leave selected the currently chosen button
        for button_label, (row, col) in self._button_positions.items():
            button = self.layout.itemAtPosition(row, col)
            if button_label == label:
                button.widget().button.setChecked(True)
            else:
                button.widget().button.setChecked(False)

    def _get_cluster_icon(self, label):
        mean_spectrogram = np.mean(self._spectrograms[self._labels == label], axis=0)
        return _spec2icon(mean_spectrogram)


class SnippetSelectView(widgets.QScrollArea):
    """A grid of pressable buttons for every cluster currently seen"""

    n_rows = 2

    def __init__(self, parent=None, cluster_selected_signal=None,
                snippet_selected_signal=None):
        super().__init__(parent)
        self._spectrograms = []

        self.init_ui()

        self.snippet_selected_signal = snippet_selected_signal
        cluster_selected_signal.connect(self.on_cluster_select)

    def init_ui(self):
        self.frame = widgets.QGroupBox()
        self.layout = widgets.QGridLayout()
        self.frame.setLayout(self.layout)
        self.setWidget(self.frame)
        self.setWidgetResizable(True)

    def reset_buttons(self):
        self._button_positions = {}
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            else:
                raise Exception("Not supposed to be anything else")

        for i, idx in enumerate(self._indices):
            row = i % self.n_rows
            col = i // self.n_rows
            cluster_select = ImgButton(
                self._get_spectrogram_icon(i),
                idx,
                self._button_callback
            )
            self._button_positions[idx] = (row, col)
            self.layout.addWidget(cluster_select, row, col, 1, 1)

    def _get_spectrogram_icon(self, idx):
        # Need to flip the freq axis
        spec = np.abs(self._spectrograms[idx])
        return _spec2icon(spec)

    def _button_callback(self, idx):
        """Report the index of the selected snippet up the chain
        """
        modifiers = widgets.QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            # select multiple mode - make the buttons pressed
            if idx in self._selected:
                self._selected.remove(idx)
            else:
                self._selected.add(idx)
        else:
            self._selected = set([idx])

        for button_idx, (row, col) in self._button_positions.items():
            button = self.layout.itemAtPosition(row, col)
            if button_idx in self._selected:
                button.widget().button.setChecked(True)
            else:
                button.widget().button.setChecked(False)

        self.snippet_selected_signal.emit(idx)

    def on_cluster_select(self, label, spectrograms, indices):
        self._spectrograms = spectrograms
        self._indices = indices
        self.reset_buttons()


class ImgButton(widgets.QFrame):
    """A pressable button that has a picture of a spectrogram on it
    """

    def __init__(self, icon, label, button_callback=None, radio=False, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._label = label
        self._button_callback = button_callback
        self._radio = radio
        self.init_ui()

        if self._button_callback is not None:
            self.button.released.connect(partial(self._button_callback, label))

    def update_icon(self):
        self.button.setIcon(self._icon)

        # TODO (kevin): fix these vlaues or configure them
        self.button.setIconSize(QtCore.QSize(60, 70))
        self.button.setFixedSize(QtCore.QSize(65, 76))
        self.setFixedSize(QtCore.QSize(100, 115))

        self.label.setText("{}".format(self._label))

    def init_ui(self):
        self.label = widgets.QLabel("n")
        if self._radio:
            self.button = widgets.QRadioButton(parent=self.parent().frame)
        else:
            self.button = widgets.QPushButton()
            self.button.setCheckable(True)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addStretch(1)
        self.setLayout(layout)
        self.update_icon()


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = App()
    window.show()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
