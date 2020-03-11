"""
Process a wav file and create a directory to store output files

Basically a script version of notebook examples 1 and 2
"""

import sys
sys.path.append("code/soundsep")

import argparse
import glob
import os
import textwrap
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
from interfaces.audio import LazyMultiWavInterface, LazyWavInterface


parser = argparse.ArgumentParser(
    prog='process_wav_file.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Process wav file
        --------------------------------
        Load a wav file for segmentation and spectrogram extraction.

        Specify either a single wav file or a folder with wav files named
        ch0.wav, ch1.wav, etc.

        In an output folder, create numpy files with extracted call
        intervals and multi-channel spectrograms.
        """))
parser.add_argument("location", type=str,
    help="Path to wav file or folder containing wav files")
parser.add_argument("-c", "--channel", type=int, default=0,
    help="Channel index to threshold on")
parser.add_argument("--canary", action="store_true",
    help="Canary Flag")


if __name__ == "__main__":
    args = parser.parse_args()

    filename = args.location
    channel = args.channel

    if not os.path.exists(filename):
        print("Could not find {}".format(filename))
        sys.exit(1)
    elif os.path.isdir(filename):
        mode = "dir"
        dirname = filename
        basename = os.path.basename(filename)
        output_folder = os.path.join(dirname, "outputs")
    elif os.path.splitext(filename)[1] == ".wav":
        mode = "wav"
        dirname = os.path.dirname(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        output_folder = os.path.join(dirname, "outputs")
    else:
        print("Filename {} must be a .wav file or a folder "
            "containing .wav files")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print("You have have run this before; the data exists at {}".format(output_folder))
        ok = input("Input [y] to overwrite existing data: ")
        if ok.lower() != "y":
            print("""

            Aborting. If you want to view the data you ran before,
            Run the jupyter notebook and open notebooks/View Detected Sounds.ipynb...
            Fill in the paths to the corresponding variables:

            SPECTROGRAMS_PATH = "{}"
            WAVFILE_PATH = "{}"
            INTERVALS_PATH = "{}"

            """.format(
                os.path.join("..", output_folder, "spectrograms.npy".format(basename)),
                os.path.join("..", filename) if mode == "wav" else os.path.join("..", dirname, "ch0.wav"),
                os.path.join("..", output_folder, "intervals.npy".format(basename)),
            ))
            sys.exit(0)
        else:
            for f in glob.glob(os.path.join(output_folder, "*.npy")):
                os.remove(f)

    if mode == "dir":
        audio_signal = LazyMultiWavInterface.create_from_directory(dirname)
    elif mode == "wav":
        audio_signal = LazyWavInterface(filename, dtype=np.float64)

    # Initial thresholding
    _t = time.time()
    print("Thresholding data...")
    if args.canary:
        all_intervals = threshold_all_events(
            audio_signal,
            channel=channel,
            window_size=3.0,
            ignore_width=0.005,
            min_size=0.005,
            fuse_duration=0.02,
            threshold_z=2.0,
        )
    else:
        all_intervals = threshold_all_events(
            audio_signal,
            channel=channel,
            window_size=10.0,
        )

    print("Detected intervals in {:.2f}s file in {:.2f}s".format(
        len(audio_signal) / audio_signal.sampling_rate,
        time.time() - _t
    ))

    if not args.canary:
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
                sig[:, channel],
                audio_signal.sampling_rate,
                expected_call_max_duration=0.1 if args.canary else 0.5,
                max_tries=10,
                scale_factor=1.25,
            )

            # since we padded the signal above, we need to be careful
            # about not accidentally adding intervals outside our original t1 -> t2
            # slice (they might belong to another detection window)
            for idx1, idx2 in inner_intervals:
                if (idx1 / audio_signal.sampling_rate) < padding:
                    if (idx2 / audio_signal.sampling_rate) < padding:
                        continue
                    else:
                        intervals.append((
                            t1,
                            t1 - padding + (idx2 / audio_signal.sampling_rate)
                        ))
                elif t1 - padding + (idx2 / audio_signal.sampling_rate) > t2:
                    if t1 - padding + (idx1 / audio_signal.sampling_rate) > t2:
                        continue
                    else:
                        intervals.append((
                            t1 - padding + (idx1 / audio_signal.sampling_rate),
                            t2
                        ))
                else:
                    intervals.append((
                        t1 - padding + (idx1 / audio_signal.sampling_rate),
                        t1 - padding + (idx2 / audio_signal.sampling_rate)
                    ))
    else:
        intervals = all_intervals

    print("Split intervals in {:.2f}s file in {:.2f}s".format(
        len(audio_signal) / audio_signal.sampling_rate,
        time.time() - _t
    ))

    ### SAVING intervals.npy
    np.save(os.path.join(output_folder, "intervals.npy"), np.array(intervals))

    centers_of_mass = []
    all_call_spectrograms = []
    all_calls = []

    _time = time.time()

    print("Extracting spectrograms from {} intervals".format(len(intervals)))
    for idx, (t1, t2) in enumerate(intervals):
        print("Working on {}/{} ({:.2f}s elapsed)".format(idx + 1, len(intervals), time.time() - _time), end="\r")

        # Recentered signal with a small buffer of 40ms on either side
        buffer = 0.01
        t_arr, sig = audio_signal.time_slice(
            max(0, t1 - buffer),
            min(audio_signal.t_max, t2 + buffer)
        )
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, audio_signal.sampling_rate, 1000, 8000).T

        amp_env = get_amplitude_envelope(sig, fs=audio_signal.sampling_rate,
                                         lowpass=8000, highpass=1000)

        # Compute the temporal center of mass of the signal
        center_of_mass = t1 - buffer + np.sum((t_arr * np.sum(amp_env, axis=1))) / np.sum(amp_env)

        # Recentered signal with a small buffer of 40ms on either side
        buffer = 0.08
        t_arr, sig = audio_signal.time_slice(
            max(0, center_of_mass - buffer),
            min(audio_signal.t_max, center_of_mass + buffer)
        )
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

    np.save(os.path.join(output_folder, "spectrograms.npy"), all_call_spectrograms)
    np.save(os.path.join(output_folder, "calls.npy"), all_calls)

    print("""

    {} potential calls detected

    Run the jupyter notebook and open notebooks/View Detected Sounds.ipynb...
    Fill in the paths to the corresponding variables:

    SPECTROGRAMS_PATH = "{}"
    WAVFILE_PATH = "{}"
    INTERVALS_PATH = "{}"

    """.format(
        len(intervals),
        os.path.join("..", output_folder, "spectrograms.npy"),
        os.path.join("..", filename) if mode == "wav" else os.path.join("..", dirname, "ch0.wav"),
        os.path.join("..", output_folder, "intervals.npy"),
    ))
