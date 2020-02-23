import numpy as np

from soundsig.signal import lowpass_filter, highpass_filter


def get_amplitude_envelope(data, fs=30000.0, lowpass=5000.0, highpass=2000.0):
    filtered = highpass_filter(data.T, fs, highpass).T

    # Rectify and lowpass filter
    filtered = np.abs(lowpass_filter(filtered.T, fs, lowpass).T)
    filtered = lowpass_filter(filtered.T, fs, 100).T

    return filtered


def interval_idx2time(intervals, sampling_rate):
    return [
        (x / sampling_rate, y / sampling_rate)
        for x, y in intervals
    ]

def interval_time2idx(intervals, sampling_rate):
    return [
        (int(x * sampling_rate), int(y * sampling_rate))
        for x, y in intervals
    ]
