import numpy as np
import soundfile


class AudioSliceInterface(object):
    """Abstract base class for data access"""
    def time_slice(self, t_start, t_stop):
        if t_start < 0.0:
            raise ValueError("t_start ({:.2f}) cannot be < 0".format(t_start))
        if t_stop > int(len(self) / self.sampling_rate):
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

    def __len__(self):
        return self._frames

    def _time_slice(self, t_start, t_stop):
        offset = int(np.round(t_start * self.sampling_rate))
        duration = int(np.round((t_stop - t_start) * self.sampling_rate))

        data, _ = soundfile.read(self._filename, duration, offset, dtype=self._dtype)
        return data


class LazySignalInterface(AudioSliceInterface):
    """Placeholder for reading from custom lazy loading object
    """
    def __init__(self, lazy_object):
        self._lazy_object = lazy_object
        self.sampling_rate = self._lazy_object.signals[0]["mic"].sampling_rate

    def _time_slice(self, t_start, t_stop):
        return self._lazy_object.time_slice(t_start, t_stop)


#  Using soundfile.SoundFile instead now since wave package doesn't seem
#    to support float64 dtype?
'''class LazyWavLoader(object):
    """Lazy .wav file loader

    Using soundfile.SoundFile instead now since wave package doesn't seem
    to support float64 dtype?
    """

    def __repr__(self):
        return "LazyWavLoader({}) <framerate={}, frames={}, channels={}>".format(
            self._filename, self._framerate, self._nframes, self._nchannels
        )

    def __init__(self, filename):
        raise NotImplementedError("Not using because wav package does not support float64 dtype.")
        self._filename = filename
        with wave.open(self._filename, "rb") as wavfile:
            self._framerate = wavfile.getframerate()
            self._nchannels = wavfile.getnchannels()
            self._sampwidth = wavfile.getsampwidth()
            self._nframes = wavfile.getnframes()

    def read(self, n_samples, n_offset=0):
        with wave.open(self._filename, "rb") as wavfile:
            wavfile.setpos(n_offset)
            s = wavfile.readframes(n_samples)

        # The i2 assumes the data is in int16 format with _sampwidth = 2
        # This can be generalized using
        #     pyaudio.get_format_from_width(_sampwidth)
        # and refering to the scipy.io.wavfile source code that selects
        # the correct dtype

        data = np.frombuffer(s, dtype="i2")
        return data.reshape(-1, self._nchannels)
'''
