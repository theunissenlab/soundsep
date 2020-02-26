"""
Process a wav file and create a directory to store output files

Basically a script version of notebook examples 1 and 2
"""

import sys
sys.path.append("code/soundsep")

import os
import time

import numpy as np
from soundsig.sound import spectrogram
from soundsig.signal import bandpass_filter

from audio_utils import get_amplitude_envelope, interval_idx2time
from detection.thresholding import (
    compute_smart_threshold,
    split_individual_events,
    threshold_all_events,
    threshold_events
)
from interfaces.audio import LazyWavInterface


if __name__ == "__main__":
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print("Could not find {}".format(filename))
        sys.exit(1)
    else:
        dirname = os.path.dirname(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        output_folder = os.path.join(dirname, "outputs", basename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    audio_signal = LazyWavInterface(filename, dtype=np.float64)

    # Initial thresholding
    _t = time.time()
    print("Thresholding data...")
    all_intervals = threshold_all_events(audio_signal, window_size=10.0)
    print("Detected intervals in {:.2f}s file in {:.2f}s".format(
        len(audio_signal) / audio_signal.sampling_rate,
        time.time() - _t
    ))

    # Split into smaller intervals
    print("Splitting {} intervals into smaller intervals".format(len(all_intervals)))
    _t = time.time()
    intervals = []
    for idx, (t1, t2) in enumerate(all_intervals):
        padding = 1.0
        if t1 - padding < 0 or t2 + padding > int(len(audio_signal) / audio_signal.sampling_rate):
            continue
        t_arr, sig = audio_signal.time_slice(t1 - padding, t2 + padding)
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        inner_intervals = split_individual_events(
            sig[:, 0],
            audio_signal.sampling_rate,
            expected_call_max_duration=0.5,
            max_tries=10,
            scale_factor=1.25,
        )

        for idx1, idx2 in inner_intervals:
            intervals.append((
                t1 - padding + (idx1 / audio_signal.sampling_rate),
                t1 - padding + (idx2 / audio_signal.sampling_rate)
            ))

    print("Split intervals in {:.2f}s file in {:.2f}s".format(
        len(audio_signal) / audio_signal.sampling_rate,
        time.time() - _t
    ))

    ### SAVING intervals.npy
    np.save(os.path.join(output_folder, "{}_intervals.npy".format(basename)), np.array(intervals))

    centers_of_mass = []
    all_call_spectrograms = []
    all_calls = []

    _time = time.time()

    print("Extracting spectrograms from {} intervals".format(len(intervals)))
    for idx, (t1, t2) in enumerate(intervals):
        print("Working on {}/{} ({:.2f}s elapsed)".format(idx + 1, len(intervals), time.time() - _time), end="\r")

        # Recentered signal with a small buffer of 40ms on either side
        buffer = 0.01
        t_arr, sig = audio_signal.time_slice(t1 - buffer, t2 + buffer)
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        amp_env = get_amplitude_envelope(sig, fs=audio_signal.sampling_rate,
                                         lowpass=8000, highpass=1000)

        # Compute the temporal center of mass of the signal
        center_of_mass = t1 - buffer + np.sum((t_arr * np.sum(amp_env, axis=1))) / np.sum(amp_env)

        # Recentered signal with a small buffer of 40ms on either side
        buffer = 0.04
        t_arr, sig = audio_signal.time_slice(center_of_mass - buffer, center_of_mass + buffer)
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        specs = []
        all_calls.append(sig)
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
    all_calls = np.array(all_calls)

    np.save(os.path.join(output_folder, "{}_spectrograms.npy".format(basename)), all_call_spectrograms)
    np.save(os.path.join(output_folder, "{}_calls.npy".format(basename)), all_calls)
