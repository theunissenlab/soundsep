import sys
import glob
import os
from functools import partial

import numpy as np
import pandas as pd
from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets
from fbs_runtime.application_context.PyQt5 import ApplicationContext

from interfaces.audio import (
    LazyMultiWavInterface,
    LazySignalInterface,
    LazyWavInterface,
    ConcatenatedMultiChannelInterface,
)

from app.state import AppState, ViewState
from app.views import MainView
from app.settings import fonts, read_default
from app.style import qss


class Events(widgets.QWidget):
    createSource = pyqtSignal(object)
    sourcesChanged = pyqtSignal()
    rangeChanged = pyqtSignal()
    dataLoaded = pyqtSignal()
    rangeSelected = pyqtSignal(object, object)
    rangeHighlighted = pyqtSignal(object, object)
    setPosition = pyqtSignal([object], [object, object])
    zoomEvent = pyqtSignal([int], [int, float], [str])
    triggerShortcut = pyqtSignal(str)
    playAudio = pyqtSignal(int)


class App(widgets.QMainWindow):
    """Main App instance with logic for file read/write
    """

    shortcut_codes = [
        "A",
        "Shift+A",
        "D",
        "Shift+D",
        "E",
        "M",
        "Q",
        "S",
        "Shift+S",
        "W",
        "Shift+W",
        "X",
        "Z",
        "Space",
        "Escape"
    ]

    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.settings = QSettings("Theuniseen Lab", "Sound Separation")
        #self.sources = SourceManager(None)
        self.state = AppState()
        self.view_state = ViewState()
        self.events = Events()

        self.init_ui()
        self.init_actions()
        self.init_menus()
        self.update_open_recent_actions()
        self.display_main()

        if self.settings.value("OPEN_RECENT", []):
            self.load_dir(self.settings.value("OPEN_RECENT")[-1])

    def init_actions(self):
        self.open_directory_action = widgets.QAction("Open Directory", self)
        self.open_directory_action.triggered.connect(self.run_directory_loader)

        self.open_recent_actions = []
        for i in range(read_default.MAX_RECENT_FILES):
            action = widgets.QAction("", self)
            action.setVisible(False)
            action.triggered.connect(partial(self.open_recent, i))
            self.open_recent_actions.append(action)

        self.load_sources_action = widgets.QAction("Load Sources", self)
        self.load_sources_action.triggered.connect(self.load_sources)

        self.save_action = widgets.QAction("Save", self)
        self.save_action.triggered.connect(self.save)
        self.save_as_action = widgets.QAction("Save As", self)
        self.save_as_action.triggered.connect(partial(self.save, save_as=True))

        self.quit_action = widgets.QAction("Close", self)
        self.quit_action.triggered.connect(self.close)
        self.export_action = widgets.QAction("Export Pickle", self)
        self.export_action.triggered.connect(partial(self.export, "pkl"))
        self.export_csv_action = widgets.QAction("Export CSV", self)
        self.export_csv_action.triggered.connect(partial(self.export, "csv"))

        self.save_shortcut = widgets.QShortcut(gui.QKeySequence.Save, self)
        self.save_shortcut.activated.connect(self.save)

        self.close_shortcut = widgets.QShortcut(gui.QKeySequence.Close, self)
        self.close_shortcut.activated.connect(self.close)

        self.help_action = widgets.QAction("Help", self)
        self.help_action.triggered.connect(self.help)

        for code in self.shortcut_codes:
            shortcut = widgets.QShortcut(gui.QKeySequence(code), self)
            shortcut.activated.connect(partial(self.pass_shortcut, code))

        # self.show_pref_action = widgets.QAction("Amplitude Envelope Parameters", self)
        # self.show_pref_action.triggered.connect(self.amp_env_pref_window.show)

    def pass_shortcut(self, shortcut):
        """
        For some reason I can't get shortcuts defined in the child widgets
        to work. So I have to define the shortcuts in this app and then pass
        them through an event.
        """
        self.events.triggerShortcut.emit(shortcut)

    def init_ui(self):
        self.setWindowTitle(self.title)

    def init_menus(self):
        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(self.open_directory_action)
        self.openRecentMenu = fileMenu.addMenu("&Open Recent")
        for i in range(read_default.MAX_RECENT_FILES):
            self.openRecentMenu.addAction(self.open_recent_actions[i])
        fileMenu.addSeparator()
        fileMenu.addAction(self.load_sources_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.export_action)
        fileMenu.addAction(self.export_csv_action)
        fileMenu.addAction(self.save_action)
        fileMenu.addAction(self.save_as_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.quit_action)

        settingsMenu = mainMenu.addMenu("&Settings")
        helpMenu = mainMenu.addMenu("&Help")
        helpMenu.addAction(self.help_action)

        # settingsMenu.addAction(self.show_pref_action)

    def help(self):
        url = QtCore.QUrl(read_default.GITHUB_LINK)
        gui.QDesktopServices.openUrl(url)

    def display_main(self):
        self.main_view = MainView(self)
        self.setCentralWidget(self.main_view)
        self.resize(1024, 768)

        self.show()

    def update_open_recent_actions(self):
        recently_opened = self.settings.value("OPEN_RECENT", [])
        for i in range(read_default.MAX_RECENT_FILES):
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
            self.load_dir(selected_file)

    def load_dir(self, dir):
        if not os.path.isdir(dir):
            widgets.QMessageBox.warning(
                self,
                "Error",
                "{} is not a directory".format(dir),
            )
            return

        # Update the open recent menu item
        open_recent = self.settings.value("OPEN_RECENT", [])
        try:
            idx = open_recent.index(dir)
        except ValueError:
            open_recent.append(dir)
        else:
            open_recent.pop(idx)
            open_recent.append(dir)

        max_recent = read_default.MAX_RECENT_FILES
        open_recent = open_recent[-max_recent:]
        self.settings.setValue("OPEN_RECENT", open_recent)
        self.update_open_recent_actions()

        self.state.reset()

        self._load_dir(dir)

    def _load_dir(self, dir):
        data_directory = os.path.join(dir, "outputs")
        wav_files = glob.glob(os.path.join(dir, "ch[0-9]*.wav"))
        wav_dir_files = glob.glob(os.path.join(dir, "ch[0-9]", "*.wav"))

        lazy_file = os.path.join(dir, "lazy.npy")
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        self.save_file = os.path.join(data_directory, "save.npy")
        if not len(wav_files) and not os.path.exists(lazy_file) and not wav_dir_files:
            widgets.QMessageBox.warning(
                self,
                "Error",
                "No wav files found.",
            )
            return

        elif not len(wav_files) and len(wav_dir_files):
            sound_object = ConcatenatedMultiChannelInterface.create_from_directory(
                dir,
                force_equal_length=True
            )
        elif os.path.exists(lazy_file):
            sound_object = LazySignalInterface(lazy_file)
            # The absolute path of the file when loaded remotely may not exist
            if not sound_object.validate_paths():
                # First try the most likely path.
                # This should work usually for songephys data...
                suggested = lazy_file.split("bird")[0]
                sound_object.set_data_path(suggested)
                if not sound_object.validate_paths():
                    # Ask the user to point to the remote drive "bird" directory
                    widgets.QMessageBox.about(
                        self,
                        "Data directory not found",
                        "Path to audio data was not found for {}. If using a remote mounted drive. "
                        "select the data directory containing the folder \"birds\"".format(lazy_file)
                    )
                    options = widgets.QFileDialog.Options()
                    data_path = widgets.QFileDialog.getExistingDirectory(
                        self,
                        "Choose the data directory containing the folder \"birds\"",
                        suggested,
                        options=options
                    )
                    if not data_path:
                        return
                    sound_object.set_data_path(data_path)
        elif len(wav_files) > 1:
            sound_object = LazyMultiWavInterface.create_from_directory(dir, force_equal_length=True)
        elif len(wav_files) == 1:
            sound_object = LazyWavInterface(wav_files[0])

        self.state.set("sources", [])

        if os.path.exists(self.save_file):
            loaded_data = np.load(self.save_file, allow_pickle=True)[()]
            if "sources" in loaded_data:
                self.state.set("sources", loaded_data["sources"])

            if "autodetect" in loaded_data:
                self.state.set("autodetected_periods", loaded_data["autodetect"])
            # if "_VIEW_STATE" in loaded_data:
            #     self.view_state.update(loaded_data["_VIEW_STATE"])

        self.state.set("sound_object", sound_object)
        self.state.set("sound_file", dir)
        self.events.dataLoaded.emit()

    def load_sources(self):
        options = widgets.QFileDialog.Options()
        file_name, _ = widgets.QFileDialog.getOpenFileName(
            self,
            "Load sources",
            self.save_file,
            "*",
            options=options)

        if file_name:
            loaded_data = np.load(file_name, allow_pickle=True)[()]
            if "sources" in loaded_data:
                self.state.set("sources", loaded_data["sources"])
                self.events.sourcesChanged.emit()

    def open_recent(self, i):
        self.load_dir(self.settings.value("OPEN_RECENT")[-i])

    def save(self, save_as=False):
        if not save_as:
            msg = ("Are you sure you want to save?\n"
                "Saving will overwrite any previously saved data.")
            reply = widgets.QMessageBox.question(
                    self,
                    'Save Confirmation',
                    msg,
                    widgets.QMessageBox.Yes,
                    widgets.QMessageBox.No)
            if reply == widgets.QMessageBox.No:
                return

        save_data = {}
        if self.state.has("sources"):
            save_data["sources"] = self.state.get("sources")
        # save_data["_VIEW_STATE"] = dict(self.view_state.__dict__)

        if save_as:
            options = widgets.QFileDialog.Options()
            save_file, _ = widgets.QFileDialog.getSaveFileName(
                self,
                "Save intervals as",
                self.save_file,
                "*",
                options=options)
        else:
            save_file = self.save_file

        np.save(self.save_file, save_data)
        widgets.QMessageBox.about(
            self,
            "Saved",
            "Saved successfully.",
        )

    def export(self, fmt="csv"):
        # Save the sources to a pandas dataframe
        if not self.state.has("sources"):
            widgets.QMessageBox.about(self, "!", "No sources to export")

        rows = []
        for source in self.state.get("sources"):
            if isinstance(source.get("intervals"), pd.DataFrame):
                df = source.get("intervals")
                for i in np.arange(len(df)):
                    t0, t1 = df.iloc[i][["t_start", "t_stop"]]
                    rows.append([source["name"], source["channel"], t0, t1])

        df = pd.DataFrame(rows, columns=["source_name", "source_channel", "t_start", "t_stop"])

        options = widgets.QFileDialog.Options()
        file_name, _ = widgets.QFileDialog.getSaveFileName(
            self,
            "Export data",
            os.path.join(self.state.get("sound_file"), "intervals.{}".format(fmt)),
            "*",
            options=options)
        if not file_name:
            return
        else:
            if fmt == "pkl":
                df.to_pickle(file_name)
            elif fmt == "csv":
                df.to_csv(file_name)
            widgets.QMessageBox.about(
                self,
                "Exported",
                "Exported successfully.",
            )



if __name__ == '__main__':
    from app.context import context
    window = App()
    window.setFont(fonts.default)
    window.setStyle(widgets.QStyleFactory.create("Fusion"))
    window.setStyleSheet(qss)
    window.show()
    exit_code = context.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
