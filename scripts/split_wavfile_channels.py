"""Split the multiple channels in one wavfile up into multiple wav files
"""

import argparse
import glob
import os
import textwrap

import scipy.io.wavfile

parser = argparse.ArgumentParser(
    prog='split_wavfile_channels.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Split Wavfile Channels
        --------------------------------
        Takes a wavfile with 2 or more channels, and saves to
        an output directory each channel in an individual file named
        ch0.wav, ch1.wav, etc.
        """))
parser.add_argument("file", type=str, help="Path to wav file")
parser.add_argument("--dest", type=str, help="Folder to save output in")


if __name__ == "__main__":
    args = parser.parse_args()

    filename = args.file
    dest = args.dest

    if not os.path.exists(dest):
        os.makedirs(dest)
    else:
        existing_files = glob.glob(os.path.join(dest, "ch*.wav"))
        if existing_files:
            print("The following files already exist: {}".format(existing_files))
            ok = input("Input [y] to overwrite existing data: ")
            if ok.lower() != "y":
                print("Aborting.")
                sys.exit(0)

    fs, data = scipy.io.wavfile.read(filename)
    n_channels = data.shape[1]

    for ch in range(n_channels):
        output_filename = os.path.join(dest, "ch{}.wav".format(ch))
        scipy.io.wavfile.write(output_filename, fs, data[:, ch])
        print("Saved {}".format(output_filename))
