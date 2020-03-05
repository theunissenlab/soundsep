import numpy as np
import soundfile


class AudioSliceInterface(object):
    """Abstract base class for data access"""

    def time_slice(self, t_start, t_stop):
        if t_start < 0.0:
            raise ValueError("t_start ({:.2f}) cannot be < 0".format(t_start))
        if t_stop > len(self) / self.sampling_rate:
            raise ValueError("t_stop ({:.2f}) cannot be > {:.2f} "
                "(the length of the file)".format(t_stop, int(len(self) / self.sampling_rate)))
        data = self._time_slice(t_start, t_stop)
        t_arr = np.arange(len(data)) / self.sampling_rate
        return t_arr, data

    def _time_slice(self, t_start, t_stop):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class NumpyDataInterface(AudioSliceInterface):
    """Read amplitude data from a numpy array
    """
    def __init__(self, data, sampling_rate):
        self._data = data
        self.sampling_rate = sampling_rate
        self.n_channels = self._data.shape[1]

    def __len__(self):
        return self._data.shape[0]

    def _time_slice(self, t_start, t_stop):
        idx1 = int(np.round(t_start * self.sampling_rate))
        idx2 = int(np.round(t_stop * self.sampling_rate))
        return self._data[idx1:idx2]


class LazyWavInterface(AudioSliceInterface):
    """Read amplitude data directly from a .wav file

    Reads the data into memory only when requested (lazy loading)

    Files used must be of specific format - if there are errors
    in loading or reading data, try re-saving the wav file using
    scipy.io.wavfile.write, and use the np.int16 dtype.
    """
    def __init__(self, filename, dtype=np.float64):
        self._dtype = dtype
        self._filename = filename
        with soundfile.SoundFile(filename) as f:
            self.sampling_rate = f.samplerate
            self._frames = f.frames
            self.n_channels = f.channels

    def __len__(self):
        return self._frames

    def _time_slice(self, t_start, t_stop):
        offset = int(np.round(t_start * self.sampling_rate))
        duration = int(np.round((t_stop - t_start) * self.sampling_rate))

        data, _ = soundfile.read(self._filename, duration, offset, dtype=self._dtype)

        if data.ndim == 1:
            data = data[:, None]

        return data

    @property
    def t_max(self):
        return self._frames / self.sampling_rate


class LazyMultiWavInterface(LazyWavInterface):
    """Read amplitude data directly from multiple wav files (1 per channel)

    Reads the data into memory only when requested

    Files must only have one channel each (enforced at read time), all
    have the same sampling rate, and all have the same length.

    Files used must be of specific format - if there are errors
    in loading or reading data, try re-saving the wav file using
    scipy.io.wavfile.write, and use the np.int16 dtype.
    """
    def __init__(self, filenames, dtype=np.float64):
        self._dtype = dtype
        self._filenames = filenames
        self.sampling_rate = None
        self._frames = None
        self.n_channels = len(filenames)

        for filename in filenames:
            with soundfile.SoundFile(filename) as f:
                if self.sampling_rate is None:
                    self.sampling_rate = f.samplerate

                if self._frames is None:
                    self._frames = f.frames

                if self.sampling_rate != f.samplerate:
                    raise Exception("All .wav files must have the same sample rate\n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self.sampling_rate,
                            filename,
                            f.samplerate
                        )
                    )

                if self._frames != f.frames:
                    raise Exception("All .wav files must have the same length\n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self._frames,
                            filename,
                            f.frames
                        )
                    )

    def _time_slice(self, t_start, t_stop):
        offset = int(np.round(t_start * self.sampling_rate))
        duration = int(np.round((t_stop - t_start) * self.sampling_rate))

        data = np.zeros((duration, self.n_channels))

        for ch, filename in enumerate(self._filenames):
            ch_data, _ = soundfile.read(filename, duration, offset, dtype=self._dtype)

            if ch_data.ndim == 1:
                ch_data = ch_data[:, None]

            if ch_data.shape[1] > 1:
                raise Exception("Cannot use .wav file with more than 1 channel\n"
                    "{} has {} channels.".format(filename, ch_data.shape[1]))

            data[:, ch] = ch_data[:, 0]

        return data


'''
class LazySignalInterface(AudioSliceInterface):
    """Placeholder for reading from custom lazy loading object
    """
    def __init__(self, lazy_object):
        self._lazy_object = lazy_object
        self.sampling_rate = self._lazy_object.signals[0]["mic"].sampling_rate

    def _time_slice(self, t_start, t_stop):
        return self._lazy_object.time_slice(t_start, t_stop)
'''
