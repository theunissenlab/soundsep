import glob
import os
import re

import numpy as np
import soundfile


class AudioSliceInterface(object):
    """Abstract base class for data access"""

    def time_slice(self, t_start, t_stop):
        if t_start < 0.0:
            raise ValueError("t_start ({:.2f}) cannot be < 0".format(t_start))
        if t_stop > len(self) / self.sampling_rate:
            raise ValueError("t_stop ({:.2f}) cannot be > {:.2f} "
                "(the length of the file)".format(t_stop, len(self) / self.sampling_rate))
        data = self._time_slice(t_start, t_stop)
        t_arr = np.arange(len(data)) / self.sampling_rate
        return t_arr, data

    @property
    def t_max(self):
        return self._frames / self.sampling_rate

    def _time_slice(self, t_start, t_stop):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def t_max(self):
        return self._frames / self.sampling_rate


class NumpyDataInterface(AudioSliceInterface):
    """Read amplitude data from a numpy array
    """
    def __init__(self, data, sampling_rate):
        self._data = data
        self.sampling_rate = sampling_rate
        self.n_channels = self._data.shape[1]
        self._frames = self._data.shape[0]

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


class LazyMultiWavInterface(LazyWavInterface):
    """Read amplitude data directly from multiple wav files (1 per channel)

    Reads the data into memory only when requested

    Files must only have one channel each (enforced at read time), all
    have the same sampling rate, and all have the same length.

    Files used must be of specific format - if there are errors
    in loading or reading data, try re-saving the wav file using
    scipy.io.wavfile.write, and use the np.int16 dtype.
    """

    @classmethod
    def create_from_directory(cls, from_directory, force_equal_length=False):
        """Convenience function to create a LazyMultiWavInterface from a
        directory containing wav files named ch0.wav, ch1.wav, ch2.wav, etc.
        """
        if not os.path.isdir(from_directory):
            raise ValueError("from_directory must be an existing directory")
        regexp = r"([0-9]+)"
        filenames = glob.glob(os.path.join(from_directory, "ch[0-9]*.wav"))
        if not len(filenames):
            raise ValueError("No filenames matching ch[0-9]*.wav were found "
                    "at {}".format(from_directory))
        filenames = sorted(filenames, key=lambda x: int(re.search(regexp, x).groups()[0]))

        return cls(filenames, force_equal_length=force_equal_length)

    def __init__(self, filenames, dtype=np.float64, force_equal_length=False):
        """

        Parameters
        ==========
        filenames:
            list of strings with file paths to wav files
        dtype: (default: np.float64)
            datatype contained in wav files
        force_equal_length: (default: False)
            if the wav files are not of the same length, will determine
            an offset (in # of samples) for each channel so that they
            match the shortest file's duration
        """
        self._dtype = dtype
        self._filenames = filenames
        self.sampling_rate = None
        self._frames = None
        self.n_channels = len(filenames)

        _channel_durations = []
        for filename in filenames:
            with soundfile.SoundFile(filename) as f:
                if self.sampling_rate is None:
                    self.sampling_rate = f.samplerate

                if self._frames is None:
                    self._frames = f.frames

                _channel_durations.append(f.frames)

                if self.sampling_rate != f.samplerate:
                    raise Exception("All .wav files must have the same sample rate\n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self.sampling_rate,
                            filename,
                            f.samplerate
                        )
                    )

                if not force_equal_length and self._frames != f.frames:
                    raise Exception("All .wav files must have the same length\n"
                        "if force_equal_length is not set to True \n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self._frames,
                            filename,
                            f.frames
                        )
                    )

        if force_equal_length:
            min_frames = np.min(_channel_durations)
            self._offsets = [frames - min_frames for frames in _channel_durations]
            self._frames = min_frames
        else:
            self._offsets = [0 for _ in _channel_durations]

    def _time_slice(self, t_start, t_stop):
        offset = int(np.round(t_start * self.sampling_rate))
        duration = int(np.round((t_stop - t_start) * self.sampling_rate))

        data = np.zeros((duration, self.n_channels))

        for ch, filename in enumerate(self._filenames):
            ch_data, _ = soundfile.read(
                filename,
                duration,
                offset + self._offsets[ch],
                dtype=self._dtype)

            if ch_data.ndim == 1:
                ch_data = ch_data[:, None]

            if ch_data.shape[1] > 1:
                raise Exception("Cannot use .wav file with more than 1 channel\n"
                    "{} has {} channels.".format(filename, ch_data.shape[1]))

            data[:, ch] = ch_data[:, 0]

        return data


class ConcatenatedWavInterface(LazyWavInterface):
    """
    Sound analysis pro records in chunks with each file timestamped.
    Works for only one subject at a time.

    This interface should read these chunks seamlessly (linking them to
    one another). The data should be organized in the filesystem like this:

    containing_folder/
        Subject1_190430_51241336.wav
        Subject1_190430_51244402.wav
        ...
    """
    def __init__(self, filenames, parse_timestamp=None, dtype=np.float64):
        self._dtype = dtype
        self._filenames = sorted(filenames, key=parse_timestamp)
        self.sampling_rate = None
        self._frames = 0        # Total number of frames across all files
        self.n_channels = None
        self._file_frames = []  # Number of frames in each file

        for filename in self._filenames:
            with soundfile.SoundFile(filename) as f:
                if self.sampling_rate is None:
                    self.sampling_rate = f.samplerate
                if self.n_channels is None:
                    self.n_channels = f.channels

                self._file_frames.append(f.frames)
                self._frames += f.frames

                if self.sampling_rate != f.samplerate:
                    raise Exception("All .wav files must have the same sample rate\n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self.sampling_rate,
                            filename,
                            f.samplerate
                        )
                    )
                if self.n_channels != f.channels:
                    raise Exception("All .wav files must have the same number of channels\n"
                        "{}: {} but {}: {}".format(
                            filenames[0],
                            self.n_channels,
                            filename,
                            f.channels
                        )
                    )

        self._file_offsets = np.cumsum(self._file_frames)  # Cumulative frames

    def _time_slice(self, t_start, t_stop, corrected_frames=None):
        offset = int(np.round(t_start * self.sampling_rate))
        duration = int(np.round((t_stop - t_start) * self.sampling_rate))

        data = np.zeros((duration, self.n_channels))

        if corrected_frames is not None:
            file_offsets = np.cumsum(corrected_frames)
        else:
            file_offsets = self._file_offsets

        start_file_idx = np.searchsorted(file_offsets, offset, side="right")

        frames_read = 0
        for file_idx in range(start_file_idx, len(self._filenames)):
            file_offset = file_offsets[file_idx - 1] if file_idx > 0 else 0

            read_start = offset + frames_read - file_offset
            read_stop = min(read_start + duration - frames_read, corrected_frames[file_idx])

            ch_data, _ = soundfile.read(
                self._filenames[file_idx],
                read_stop - read_start,
                read_start,
                dtype=self._dtype
            )

            if ch_data.ndim == 1:
                ch_data = ch_data[:, None]

            data[frames_read:frames_read + ch_data.shape[0]] = ch_data
            frames_read += ch_data.shape[0]
            if frames_read == duration:
                break

        return data


class ConcatenatedMultiChannelInterface(LazyWavInterface):
    """Concatenate data across channels and time

    Data should be organized like this:
    containing_folder/
        ch0/
            Subject1_190430_51241336.wav
            Subject1_190430_51244402.wav
            ...
        ch1/
            Subject2_190430_51241336.wav
            Subject2_190430_51244402.wav
            ...

    And read like
    ConcatenatedMultiChannelInterface.create_from_directory(
        "containing_folder"
    )
    """

    @classmethod
    def create_from_directory(cls, from_directory):
        """Convenience function to create a LazyMultiWavInterface from a
        directory containing wav files named ch0.wav, ch1.wav, ch2.wav, etc.
        """
        if not os.path.isdir(from_directory):
            raise ValueError("from_directory must be an existing directory")
        regexp = r"([0-9]+)"
        filenames = glob.glob(os.path.join(from_directory, "ch[0-9]"))
        if not len(filenames):
            raise ValueError("No folders matching ch[0-9] were found "
                    "at {}".format(from_directory))
        filenames = sorted(filenames, key=lambda x: int(re.search(regexp, x).groups()[0]))

        return cls(filenames)

    def __init__(self, foldernames, parse_timestamp=None, dtype=np.float64):
        self._dtype = dtype
        self._foldernames = foldernames
        self.sampling_rate = None
        self._frames = None        # Total number of frames across all files
        self.n_channels = len(foldernames)

        self._loaders = []

        _channel_durations = []

        for folder in self._foldernames:
            files = glob.glob(os.path.join(folder, "*.wav"))
            loader = ConcatenatedWavInterface(files, parse_timestamp=parse_timestamp, dtype=dtype)
            if loader.n_channels > 1:
                raise Exception("Wav files loaded by ConcatenatedMultiChannelInterface must have one channel each")
            self._loaders.append(loader)

            if self.sampling_rate is None:
                self.sampling_rate = loader.sampling_rate

            if self._frames is None:
                self._frames = loader._frames

            _channel_durations.append(loader._frames)

        # if the loaders don't all have the same lenth of files
        if len(set([len(loader._file_frames) for loader in self._loaders])) > 1:
            raise Exception("All loaders in ConcatenatedMultiChannelInterface "
                "must have same number of contained files")

        self._offsets = []
        for file_idx in np.arange(len(self._loaders[0]._file_frames)):
            file_lengths = [
                loader._file_frames[file_idx]
                for loader in self._loaders
            ]
            min_frames = np.min(file_lengths)
            self._offsets.append(min_frames)

    def _time_slice(self, t_start, t_stop):
        results = []
        for i, loader in enumerate(self._loaders):
            data = loader._time_slice(
                t_start,
                t_stop,
                corrected_frames=self._offsets
            )
            results.append(data)

        return np.hstack(results)


class LazySignalInterface(AudioSliceInterface):
    """Placeholder for reading from custom lazy loading object
    """
    def __init__(self, lazy_path):
        self._path = lazy_path
        self._lazy_object = np.load(self._path, allow_pickle=True)[()]
        self.sampling_rate = self._lazy_object._signals["mic"][0].sampling_rate
        self._frames = int(self._lazy_object.max_time() * self.sampling_rate)
        self.n_channels = self._lazy_object._signals["mic"][0].M

    def _time_slice(self, t_start, t_stop):
        result = self._lazy_object.time_slice(t_start, t_stop, signal="mic")
        return result.data
