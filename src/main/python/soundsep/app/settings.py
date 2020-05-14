from PyQt5.QtCore import QSettings
from PyQt5 import QtGui as gui


class _defaults:
    MAX_RECENT_FILES = 5
    SPEC_SAMPLE_RATE = 500
    SPEC_FREQ_SPACING = 100
    MIN_FREQ = 250
    MAX_FREQ = 8000

    LORES_SCALES = [2.0, 4.0, 8.0, 12.0, 16.0]
    LORES_SAMPLE_RATE_FACTOR = [4, 12, 18, 22, 25]
    LORES_FREQ_SPACING_FACTOR = [2, 2, 2, 2, 2]

    MAX_WINDOW_SIZE = 20.0
    MIN_WINDOW_SIZE = 1.0
    WINDOW_SIZE = 6.0
    PAGE_STEP = 40  # number of pages per window


class _SettingsReader:
    """Just a helper to make reading settings or default easier to write"""
    def __init__(self, settings):
        self._settings = settings

    def __getattr__(self, key):
        default = getattr(_defaults, key)
        return type(default)(self._settings.value(key, default))

    def set(self, key, val):
        self._settings.setValue(key, val)


read_default = _SettingsReader(QSettings("Theuniseen Lab", "Sound Separation"))


class fonts:
    default = gui.QFont("Helvetica", 10)
    subheading = gui.QFont("Helvetica", 10, weight=gui.QFont.Bold)
