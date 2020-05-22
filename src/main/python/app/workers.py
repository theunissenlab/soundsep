from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot)

from soundsig.sound import spectrogram

from detection.thresholding import threshold_all_events, fuse_events


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


class VocalPeriodsWorker(QObject):
    """
    Async worker that looks for vocal periods in a selected
    time range and then returns them
    """
    finished = pyqtSignal(object, object)

    def __init__(self, key, channels, audio_signal, *args, **kwargs):
        super().__init__()
        self.key = key
        self.channels = channels
        self.audio_signal = audio_signal
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def compute(self):
        all_events = []
        for channel in self.channels:
            events = threshold_all_events(
                    self.audio_signal,
                    window_size=20.0,
                    channel=channel,
                    t_start=None,
                    t_stop=None,
                    ignore_width=0.01,
                    min_size=0.1,
                    fuse_duration=2.0,
                    threshold_z=2.0,
                    amp_env_mode="broadband"
            )
            all_events += events

        events = fuse_events(all_events)

        self.finished.emit(self.key, events)
