import numpy as np
import scipy.signal

from soundsig.signal import lowpass_filter, highpass_filter
from soundsig.sound import spectrogram


def get_amplitude_envelope(
            data,
            fs=30000.0,
            lowpass=8000.0,
            highpass=1000.0,
            rectify_lowpass=600.0,
            spec_sample_rate=200.0,
            spec_freq_spacing=200.0,
            mode="broadband"
        ):
    """Compute an amplitude envelope of a signal

    This can be done in two modes: "broadband" or "max_zscore"

    Broadband mode is the normal amplitude envelope calcualtion. In broadband
    mode, the signal is bandpass filtered, rectified, and then lowpass filtered.

    Max Zscore mode is useful to detect calls which may be more localized in
    the frequency domain from background (white) noise. In max_zscore mode, a
    spectrogram is computed with spec_sample_rate time bins and
    spec_freq_spacing frequency bins. For each time bin, the power in each
    frequency bin is zscored and the max zscored value is assigned to the
    envelope for that bin.
    """
    spectral = True
    filtered = highpass_filter(data.T, fs, highpass, filter_order=10).T
    filtered = lowpass_filter(filtered.T, fs, lowpass, filter_order=10).T

    if mode == "max_zscore":
        t_spec, f_spec, spec, _ = spectrogram(
            filtered,
            fs,
            spec_sample_rate=spec_sample_rate,
            freq_spacing=spec_freq_spacing,
            cmplx=False
        )
        spec = np.abs(spec)
        std = np.std(spec, axis=0)
        zscored = (spec - np.mean(spec, axis=0)) / std
        filtered = np.max(zscored, axis=0)
        filtered -= np.min(filtered)
        return scipy.signal.resample(filtered, len(data))
    elif mode == "broadband":
        # Rectify and lowpass filter
        filtered = np.abs(filtered)
        filtered = lowpass_filter(filtered.T, fs, rectify_lowpass).T
        return filtered
    else:
        raise ValueError("Invalid amp_env_mode {}".format(amp_env_mode))


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
