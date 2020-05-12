from PyQt5 import QtGui as gui


class defaults:
    MAX_RECENT_FILES = 5
    SPEC_SAMPLE_RATE = 1000
    SPEC_FREQ_SPACING = 50
    MIN_FREQ = 250
    MAX_FREQ = 8000

    LORES_SAMPLE_RATE_FACTOR = 25
    LORES_FREQ_SPACING_FACTOR = 5

    WINDOW_SIZE = 6.0
    PAGE_STEP = 40


def read_default(settings, key):
    return settings.value(key, getattr(defaults, key))


class fonts():
    default = gui.QFont("Helvetica", 10)
    subheading = gui.QFont("Helvetica", 10, weight=gui.QFont.Bold)
