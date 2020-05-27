# Documentation

## Overall organization

The main code for the gui is located in `src/main/python/soundsep/app` and the detection/threshold functions are found in `src/main/python/soundsep/detection`. `src/main/python/soundsep/interfaces` is where you would put functions that read audio files of different formats.

Default values for settings and constant values for the gui are defined in `src/main/python/soundsep/app/settings.py`. User adjusted values can/will be set in a settings file accessible with `PyQt5.QtCore.QSettings("Theuniseen Lab", "Sound Separation")` (the location of the settings ini file in the filesystem varies by operating system).

## Application State

Application state is stored in global, singleton objects that can be
instantiated multiple times but share the same underlying data. These objects
are `app.state.AppState` and `app.state.ViewState`.

### `app.state.AppState`
Stores application state. Persists when the application is reloaded.

#### Methods

`.reset()`: Reset all keys

`.set(key, value)`: Set a state value

`.get(key, default)`: Get a key. If it doesn't exist, use default

`.has(key)`: Returns bool indicating the key is set in the state

`.clear(key)`: Remove a key from the state

#### Keys

* "sound_object": Instance of subclass of `interfaces.audio.AudioSliceInterface` - this is the loaded audio data

* "sound_file": Path to directory containing the loaded data in "sound_object" - used for reloading

* "sources": List of dictionaries of sources. A Source is represented by a dictionary with the keys "name", "channel", "hidden", and sometimes "intervals". The intervals key if it exists points to a pandas DataFrame object with columns "t_start" and "t_stop" delimiting vocalization periods from that source.

* "autosearch": Is set when data selection should automatically trigger the vocalization period detection function. Deleted when this this functionality should be turned off.

### `app.state.ViewState`
Stores state of current audio view including visible time range information and highlighted data, etc.

#### Methods

`.reset()`: Reset all keys

`.set(key, value)`: Set a state value

`.get(key, default)`: Get a key. If it doesn't exist, use default

`.has(key)`: Returns bool indicating the key is set in the state

`.clear(key)`: Remove a key from the state

#### Keys

* "current_range"

* "source_focus": Is set to the channel that a selection is made in. Activates keyboard shortcuts to apply to this source when set.

* "selected_range"

* "selected_threshold_line"

* "selected_spec_range"

* "show_ampenv"

* "highlighted_range": A single highlighted range for the user. Highlight is visually separate from labeled intervals.

## API

TBD


## Building the Installer

To build the installer on ubuntu install ruby and fpm (see instructions: https://fpm.readthedocs.io/en/latest/installing.html). Then the command `fbs freeze` creates the frozen version of the app, and `fbs installer` should create the installer file `targets/SoundSep.deb` (on ubuntu).

On Mac, there are some issues more issues that make things more complicated. This is due to (I think) a pyinstaller bug with Tk and for some reason the freeze command doesn't copy one portaudio library. This is the solution I found:

```
fbs freeze  # creates app at targets/Soundsep.app but won't run
cd target/Soundsep.app/Contents/MacOS
mkdir tcl tk
mkdir _sounddevice_data
mkdir _sounddevice_data/portaudio-binaries

cp -R /Library/Frameworks/Python.framework/Versions/3.7/lib/tcl* tcl/
cp -R /Library/Frameworks/Python.framework/Versions/3.7/lib/tk* tk/
cp -R /Library/Frameworks/Python.framework/Versions/3.7/lib/Tk* tk/
cp ../../../../env/lib/python3.7/site-packages/_sounddevice_data/portaudio-binaries/libportaudio.dylib  _sounddevice_data/portaudio-binaries/

cd ../../../..
fbs installer
```

The dmg will be created in `targets/SoundSep.dmg`
