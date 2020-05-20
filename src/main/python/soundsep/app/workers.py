from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot)

from soundsig.sound import spectrogram


class SpectrogramWorker(QObject):
    """Async worker for computing spectrogram

    Computes the spectrogram for each channel requested
    and returns them in a dictionary of (channel: (t_spec, f_spec, spec))
    pairs.
    """
    finished = pyqtSignal(object, object)

    def __init__(self, key, channels, data, *args, **kwargs):
        super().__init__()
        self.key = key
        self.channels = channels
        self.data = data
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def compute(self):
        result = {}
        for ch in self.channels:
            t_spec, f_spec, spec, _ = spectrogram(self.data[:, ch], *self.args, **self.kwargs)
            result[ch] = (t_spec, f_spec, spec)
        self.finished.emit(self.key, result)
