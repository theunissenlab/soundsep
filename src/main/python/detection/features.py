import time

import numpy as np
from soundsig.signal import bandpass_filter
from soundsig.sound import spectrogram

from audio_utils import get_amplitude_envelope


def extract_spectrograms(
        audio_signal,
        intervals,
        buffer=0.02,  # buffer around original interval to keep
        spec_buffer=0.04,  # buffer so that everything thas the right padding.
    ):
    """Extract spectrograms from audio_signal denoted by intervals list

    intervals is a list of (t_start, t_end) tuples
    """
    _time = time.time()
    print("Extracting spectrograms from {} intervals".format(len(intervals)))
    # all_calls = []
    all_call_spectrograms = []
    for idx, (t1, t2) in enumerate(intervals):
        print("Working on {}/{} ({:.2f}s elapsed)".format(idx + 1, len(intervals), time.time() - _time), end="\r")

        # Recentered signal with a small buffer of 40ms on either side
        t_arr, sig = audio_signal.time_slice(
            max(0, t1 - buffer),
            min(audio_signal.t_max, t2 + buffer)
        )
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        amp_env = get_amplitude_envelope(sig,
            fs=audio_signal.sampling_rate,
            lowpass=8000,
            highpass=1000
        )

        # Compute the temporal center of mass of the signal
        center_of_mass = t1 - buffer + np.sum((t_arr * np.sum(amp_env, axis=1))) / np.sum(amp_env)

        # Recentered signal with a small buffer of 40ms on either side
        t_arr, sig = audio_signal.time_slice(
            max(0, center_of_mass - spec_buffer),
            min(audio_signal.t_max, center_of_mass + spec_buffer)
        )
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        specs = []
        # all_calls.append(sig)
        for ch in range(sig.shape[1]):
            # Sligtly lower resolution on the spectrograms can make this go faster
            # Can increase the params to 1000, 50 for a higher resolution spectrogram
            _, _, spec, _ = spectrogram(
                sig[:, ch],
                audio_signal.sampling_rate,
                500,
                100,
                min_freq=1000,
                max_freq=8000,
                cmplx=False
            )
            specs.append(spec)

        all_call_spectrograms.append(np.array(specs))

    all_call_spectrograms = np.array(all_call_spectrograms)
    # all_calls = np.array(all_calls)
    return all_call_spectrograms
