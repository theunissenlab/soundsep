import logging
import os
import pickle

import numpy as np
from numpy.lib.format import open_memmap


def time_to_index(t, sampling_rate):
    """Convert a time to an index"""
    return int(round(t * sampling_rate))


class LoadedTimeSliceDict(dict):
    """A dict with convenient string representation for time slices
    Use it like a dictionary, but initialize it with
    parameters t0 and t1. These are displayed when printing the
    object, along with any keys.
    >>> time_slice_dict = lazy_site.time_slice(0.0, 1.0)
    >>> print(time_slice_dict)
    LoadedTimeSliceDict(keys=['filtered', 'mic'], t in [0.00s, 1.00s])
    >>> print(time_slice_dict["filtered"])
    LoadedTimeSlice(shape=(30000, 16), t in [0.00s, t1=1.00s])
    """

    def __init__(self, t0, t1, *args, **kwargs):
        self._t0 = t0
        self._t1 = t1
        super().__init__(self, *args, **kwargs)

    def __repr__(self):
        return "LoadedTimeSliceDict(keys={}, t in [{:.2f}s, {:.2f}s])".format(
            list(self.keys()),
            self._t0,
            self._t1)


class LoadedTimeSlice(object):
    """Hold data and its associated time axis
    >>> time_slice = LoadedTimeSlice(...)
    >>> print(time_slice)
    LoadedTimeSlice(shape=(30000, 2), t in [0.00s, 1.00s])
    >>> print(time_slice.t)
    array([0.00000000e+00, 3.33333333e-05, 6.66666667e-05, ...,
           9.99900000e-01, 9.99933333e-01, 9.99966667e-01])
    >>> print(time_slice.data)
    memmap([[ ... ]])
    """

    def __init__(self, t, data):
        """Create a LoadedTimeSlice object
        :param t: Length N time array
        :type t:
        Args:
            t: 1d array of length N (number of samples)
            representing the time in seconds of the data
            data: 2d array of shape (N, M), with M = number of channels
        """
        self.t = t
        self.data = data

    def __repr__(self):
        return "LoadedTimeSlice(shape={}, t in [{:.2f}s, {:.2f}s])".format(
            self.data.shape,
            self.t[0],
            self.t[-1])


class LazySignal(object):
    """Access a signal with lazy loading by time slices
    Attributes
    ----------
    N : int
        size of signal in samples
    M : int
        number of channels in signal
    file : str
        path to underlying memmap file
    sampling_rate : int or float
        sampling rate in Hz
    dtype : str
        datatype of memmap file
    metadata : dict
        metadata of signal
    channel_names : list
        optional channel names of length M
    Methods
    -------
    time_slice(t_start, t_stop)
        Load data within a time range into memory
    write(self, new_array, n=None, offset=0, batch_size=None)
        Write data to disk at a particular position
    batches(step)
        Create an iterator to load one chunk of data into memory
        at a time
    >>> sig = LazySignal(
    ...     N=162000000,
    ...     file="<path>",
    ...     sampling_rate=30000.0,
    ... )
    >>> print(sig)
    LazySignal(N=162000000 <5400.0s>, file=<abs-path>, sampling_rate=30000.0)
    >>> electrode_data = sig.time_slice(30.0, 30.1)
    >>> plt.plot(electrode_data.t, electrode_data.data[:, 0])
    """

    def __init__(
            self,
            N,
            M,
            file,
            sampling_rate,
            dtype="float32",
            channel_names=None,
            metadata=None
    ):
        """Initialize the LazySignal reader
        Args
        ----
        N : int
            size of signal in samples
        M : int
            number of channels in signal
        file : str
            path to underlying memmap file
        sampling_rate : int or float
            sampling rate in Hz
        Kwargs
        ------
        dtype : str (default="float32")
            datatype of memmap file
        channel_names : list (default=[])
            optional channel names of length M
        metadata : dict (default={})
            metadata of signal
        """
        self.N = N
        self.M = M
        self.dtype = dtype
        self.file = os.path.abspath(file)
        self.sampling_rate = sampling_rate
        self.metadata = {} if not metadata else metadata

        if channel_names is not None and len(channel_names) != self.M:
            raise ValueError("LazySignal channel_names must be same length as"
                             " number of channels, M")
        self.channel_names = (
            np.arange(self.M) if not channel_names else channel_names
        )

        if self.dtype != "float32":
            raise Exception("I haven't accounted for non-float32 data yet")

        if not os.path.exists(self.file):
            placeholder = open_memmap(
                self.file,
                mode="w+",
                dtype=self.dtype,
                shape=(self.N, self.M))
            del placeholder

    def __repr__(self):
        return "LazySignal(N={} <{:.1f}s>, file={}, sampling_rate={})".format(
            self.N, self.N / self.sampling_rate, self.file, self.sampling_rate)

    def _load_samples(self, n, offset, mode="r"):
        """Load subset of array into memory
        Loads a numpy memmmap object of 32-bit floating points numbers
        into memory by indexing into the saved data array.
        Can be used to read and write depending on the current mode.
        Args
        ----
        n : int
            number of samples to load
        offset : int
            sample to start reading at
        Kwargs
        ------
        mode : str
            read or write mode
        Returns a numpy.memmap object
        """
        return np.memmap(self.file,
                         mode=mode,
                         dtype="float32",
                         shape=(n, self.M),
                         offset=int(offset * self.M * (32 // 8))
                         )

    def read_samples(self, n, offset):
        """Load read-only data into memory
        Args
        ----
        n : int
            Number of samples to load
        offset : int
            Position in array to load at
        Returns a numpy array with the requested data
        """
        memmap_file = self._load_samples(n, offset, mode="r")
        data = np.array(memmap_file)
        del memmap_file
        return data

    def _write(self, new_array, n=None, offset=0):
        if len(new_array) != self.N and n is None:
            raise ValueError("Cannot write to subset of array without specifying n")
        if offset + len(new_array) > self.N:
            raise ValueError("Array update of size {} at location {} "
                             "would extend beyond original size {}".format(len(new_array), offset, self.N))

        arr = self._load_samples(new_array.shape[0], offset, mode="r+")
        arr[:] = new_array[:]
        del arr

    def write(self, new_array, n=None, offset=0, batch_size=None):
        """(Over)write a section of the numpy array at a particular location
        Provides the ability to write in batches to avoid a single
        huge write operation
        Args
        ----
        new_array : numpy array
            The array of size n to write to disk.
        Kwargs
        ------
        n : int (default=None)
            The length of the new array being written. This can be left as
            None if the entire array (size N) is being written
        offset : int (default=0)
            The position at which the new array will be written
        batch_size : int (default=None)
            Size of chunks to write at a time, leave as None to write
            as one big chunk
        """
        if batch_size is None:
            self._write(new_array, n=n, offset=offset)
        else:
            for i in np.arange(0, len(new_array), batch_size):
                chunk = new_array[i:i + batch_size]
                self._write(chunk, n=chunk.shape[0], offset=offset + i)

    def time_slice(self, t_start, t_stop):
        """Access a subset of the data array
        Args
        ----
        t_start : float
            Time in seconds of the first sample, rounded down
        t_stop : float
            Time in seconds of the last sample, rounded down
        Returns a LoadedTimeSlice object with two attributes:
            t: 1d array of length N (number of samples) with the time in
                seconds of each sample
            data: 2d array of shape (N, M), with M the number of channels
        >>> signal = LazySignal(...)
        >>> elec = signal.time_slice(30.0, 30.1)
        >>> plt.plot(elec.t, elec.data[:, 0])
        """
        start_idx = time_to_index(t_start, self.sampling_rate)
        stop_idx = start_idx + int(round((t_stop - t_start) * self.sampling_rate))
        return self.index_slice(start_idx, stop_idx)

    def index_slice(self, start_idx, stop_idx):
        n = stop_idx - start_idx

        # If you request past the end of the slice, just pad with zeros
        # The calling function should deal with completing the rest of the slice
        t_arr = np.arange(start_idx, stop_idx) / self.sampling_rate

        if start_idx > self.N:
            return LoadedTimeSlice(
                t=t_arr,
                data=np.zeros((n, self.M)))

        pad_zeros_before = max(0 - start_idx, 0)
        pad_zeros_after = max(stop_idx - self.N, 0)

        data = self.read_samples(
            n - (pad_zeros_after + pad_zeros_before),
            start_idx + pad_zeros_before
        )

        data = np.vstack([
            np.zeros((pad_zeros_before, data.shape[1])),
            data,
            np.zeros((pad_zeros_after, data.shape[1]))
        ])

        return LoadedTimeSlice(t=t_arr, data=data)

    def batches(self, step):
        """Create a generator that iterates over the data
        Args
        ----
        step : int
            Size in samples of each batch
        """
        for start_idx in range(0, self.N, step):
            yield start_idx, self.read_samples(min(step, self.N - start_idx), start_idx)


class LazySite(object):
    """Access a signal with lazy loading by time slices
    Attributes
    ----------
    signal_times : list
        Relative start times (in seconds) of each enclosed LazySignal
    _signals : list
        List of LazySignal objects to access
    Methods
    -------
    time_slice(t_start, t_stop, signal=None)
        Load data within a time range into memory
    max_time()
        Get the maximum time of the last signal
    add_signals(signal_name, signals)
        Add a new signal by key to the site
    Example: Load the raw signals and associate with their
        relative start times
    >>> site = LazySite([ ... raw signals ...], signal_times=[0, 5400.0, 10800.0])
    >>> print(site)
    LazySite(3 signals, times=[   0. 5400. 10800.0])
    >>> plt.plot(t_arr, elec_data[:, 0])
    """

    def __init__(self, signal_times, **lazy_signals):
        """Initialize the LazySite
        Args
        ----
        signal_times : list
            Relative start times (in seconds) of each enclosed LazySignal.
            Length K corresponds to number of LazySignals
        Kwargs
        ------
        **lazy_signals : lists of LazySignal objects
            A dict of signal names (i.e. "raw", "mic", "filtered") mapping
            to lists of K LazySignal objects, one for each session,
            corresponding to the start times in signal_times argument
        """
        self._signals = lazy_signals
        self.signal_times = np.array(signal_times)

    def __repr__(self):
        return "LazySite(keys={}, {} signals, times={})".format(
            list(self._signals.keys()),
            len(self.signal_times),
            self.signal_times
        )

    def _get_lazy_signals(self, t_start):
        """Get the first signal containing time t_start
        First, find the first session that contains t_start using
        self.signal_times. Then, grab the corresponding LazySignal
        for that session from each of the keys in self._signals.
        Args
        ----
        t_start : float
            Start time in seconds to search for
        Returns a tuple of the start time of the session in seconds (float),
            and a dictionary mapping signal string names (e.g. "filtered") to
            the LazySignal object for that session.
        """
        session_idx = np.where(self.signal_times <= t_start)[0][-1]

        signals = dict((key, val[session_idx]) for key, val in self._signals.items())
        return self.signal_times[session_idx], signals

    def max_time(self):
        """Return the largest timestamp in seconds across all signals"""
        N = 0
        max_duration = 0.0
        for sigs in self._signals.values():
            if sigs[-1].N > N:
                N = sigs[-1].N
                max_duration = N / sigs[-1].sampling_rate
        return self.signal_times[-1] + max_duration

    def add_signals(self, signal_name, signals):
        """Add a new signal to the LazySite
        Args
        ----
        signal_name : str
            Name of the new signal to add
        signals : list of LazySignal objects
            LazySignal objects for each session, corresponding to
            self.signal_times
        """
        if len(signals) != len(self.signal_times):
            raise ValueError("Number of new signals ({}) does not match number of "
                             "sessions in LazySite ({}, signal_times={})".format(
                len(signals),
                len(self.signal_times),
                self.signal_times))

        if signal_name in self._signals:
            raise ValueError("Signal name {} already exists".format(signal_name))

        self._signals[signal_name] = signals

    def time_slice(self, t_start, t_stop, signal=None):
        """Access a subset of the data array by time
        Args
        ----
        t_start : float
            Time in seconds of the first sample, rounded down
        t_stop : float
            Time in seconds of the last sample, rounded down
        Kwargs
        ------
        signal : str (default=None)
            String name of the specific signal to load (e.g. "filtered",
            "mic", "raw"). If left as None, load all the signals into a
            LoadedTimeSliceDict. Specify this to avoid loading unneeded
            arrays.
        If signal is not provided, returns a LoadedTimeSliceDict whose keys
            are the signal names and values are LoadedTimeSlice objects.
            If signal is specified, directly return the LoadedTimeSlice object
            for that array.
        >>> site = LazySite(...)
        >>> site.time_slice(0.0, 1.0)
        LoadedTimeSlices(keys=['filtered', 'mic'], t in [0.00s, 1.00s])
        >>> site.time_slice(0.0, 1.0, signal="mic")
        LoadedTimeSlice(shape=(30000, 2), t in [0.00s, t1=1.00s])
        """
        t_signal, lazy_signals = self._get_lazy_signals(t_start)
        t_signal_leftover, lazy_signals_leftover = self._get_lazy_signals(t_stop)

        # Floating point errors is worst thing in the world. Must convert

        if signal is not None:
            result = lazy_signals[signal].time_slice(
                t_start - t_signal,
                t_stop - t_signal)
            result.t += t_signal

            if t_signal != t_signal_leftover:
                leftover_signal = lazy_signals_leftover[signal].time_slice(
                    t_start - t_signal_leftover,
                    t_stop - t_signal_leftover,
                )
                leftover_signal.t += t_signal_leftover
                result.data[result.t >= t_signal_leftover] = leftover_signal.data[
                    leftover_signal.t >= t_signal_leftover]

            return result

        signals = LoadedTimeSliceDict(t_start, t_stop)
        for key, lazy_signal in lazy_signals.items():
            signals[key] = lazy_signal.time_slice(t_start - t_signal, t_stop - t_signal)
            signals[key].t += t_signal  # Offset by the start time of the session

            if t_signal != t_signal_leftover:
                leftover_signal = lazy_signals_leftover[key].time_slice(
                    t_start - t_signal_leftover,
                    t_stop - t_signal_leftover,
                )
                leftover_signal.t += t_signal_leftover
                signals[key].data[signals[key].t >= t_signal_leftover] = leftover_signal.data[
                    signals[key].t >= t_signal_leftover]

        return signals

