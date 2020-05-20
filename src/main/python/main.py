import sys
sys.path.append("code/soundsep")

import glob
import os
from functools import partial

import numpy as np
from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets
from fbs_runtime.application_context.PyQt5 import ApplicationContext

from interfaces.audio import LazyMultiWavInterface, LazyWavInterface

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
    zoomEvent = pyqtSignal([int], [int, float])
    triggerShortcut = pyqtSignal(str)


class App(widgets.QMainWindow):
    """Main App instance with logic for file read/write
    """

    shortcut_codes = [
        "A",
        "Shift+A",
        "D",
        "Shift+D",
        "E"
        "M",
        "Q",
        "W",
        "Shift+W",
        "X",
        "Z",
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

        self.save_action = widgets.QAction("Save", self)
        self.save_action.triggered.connect(self.save)
        self.quit_action = widgets.QAction("Close", self)
        self.quit_action.triggered.connect(self.close)

        self.save_shortcut = widgets.QShortcut(gui.QKeySequence.Save, self)
        self.save_shortcut.activated.connect(self.save)

        self.close_shortcut = widgets.QShortcut(gui.QKeySequence.Close, self)
        self.close_shortcut.activated.connect(self.close)

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
        fileMenu.addAction(self.save_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.quit_action)

        settingsMenu = mainMenu.addMenu("&Settings")
        # settingsMenu.addAction(self.show_pref_action)

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
            raise IOError("{} is not a directory".format(dir))

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
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        self.save_file = os.path.join(data_directory, "save.npy")

        if not len(wav_files):
            widgets.QMessageBox.warning(
                self,
                "Error",
                "No wav files found.",
            )
            return
        if len(wav_files) > 1:
            sound_object = LazyMultiWavInterface.create_from_directory(dir)
        else:
            sound_object = LazyWavInterface(wav_files[0])

        self.state.set("sources", [])

        if os.path.exists(self.save_file):
            loaded_data = np.load(self.save_file, allow_pickle=True)[()]
            if "sources" in loaded_data:
                self.state.set("sources", loaded_data["sources"])
            # if "_VIEW_STATE" in loaded_data:
            #     self.view_state.update(loaded_data["_VIEW_STATE"])

        self.state.set("sound_object", sound_object)
        self.state.set("sound_file", dir)
        self.events.dataLoaded.emit()

    def open_recent(self, i):
        self.load_dir(self.settings.value("OPEN_RECENT")[-i])

    def save(self):
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

        np.save(self.save_file, save_data)
        widgets.QMessageBox.about(
            self,
            "Saved",
            "Saved successfully.",
        )


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = App()
    window.setFont(fonts.default)
    window.setStyleSheet(qss)
    window.show()
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
