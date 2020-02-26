# Sound Separation

Sound separation code for zebra finch vocalizations.

## Extract potential calls from a file

This is a quick script that extracts calls from the first channel of a wav file. From top level of the project (after installing dependencies and activating virtual environment), run

```
python scripts/process_wav_file.py <path_to_wav_file>
```

This creates a directory in the same folder as the wav file called "output" where the output numpy files will be saved.


## Set up for development

Installation of QT Creator (only required for designing GUI with a GUI)
```
sudo apt-get install qttools5-dev-tools
```

Creation of virtual environment for python dependencies
```
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```

Running the GUI
```
fbs run
```
