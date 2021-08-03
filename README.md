# This is an old version

Use the new version at https://github.com/theunissenlab/soundsep2


# Sound Separation

Sound separation code for zebra finch vocalizations.

## Initial setup

Creation of virtual environment for python dependencies
```
virtualenv env -p python3
```

Activation of virtual environment
```
source env/bin/activate
```

Install python dependencies
```
pip install -r requirements.txt
```

Create code -> src/main/python/ symlink
```
ln -s src/main/python code
```

#### On Windows

Installation of hdbscan requires Microsoft Visual C++ 14.0 or greater (get Microsoft C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/). hdbscan might not be necessary though (used for clustering) so you could comment out the line in requirements.txt intsead.

From the terminal
```
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
mklink /D code src\main\python
```

## Extract potential calls from a file

This is a quick script that extracts calls from the first channel of a wav file. A more sophisticated extraction strategy from multichannel wav files or multiple simultaneously recorded wav files should use custom code based on this script.

From top level of the project (after installing dependencies and activating virtual environment), run

```
python scripts/process_wav_file.py <path_to_wav_file>
```

This creates a directory in the same folder as the wav file called "output" where the output numpy files will be saved.

## Other setup options

Installation of QT Creator (only for GUI designer)
```
sudo apt-get install qttools5-dev-tools
```

Running the GUI
```
fbs run
```
