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

## Extract potential calls from a file

This is a quick script that extracts calls from the first channel of a wav file. A more sophisticated extraction strategy from multichannel wav files or multiple simultaneously recorded wav files should use custom code based on this script.

From top level of the project (after installing dependencies and activating virtual environment), run

```
python scripts/process_wav_file.py <path_to_wav_file>
```

This creates a directory in the same folder as the wav file called "output" where the output numpy files will be saved.

## Other setup options

Installation of QT Creator (only required for designing GUI with a GUI)
```
sudo apt-get install qttools5-dev-tools
```

Running the GUI (not made yet)
```
fbs run
```
