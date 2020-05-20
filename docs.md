# Documentation

## Application State

Application state is stored in global, singleton objects that can be
instantiated multiple times but share the same underlying data. These objects
are `app.state.AppState` and `app.state.ViewState`.

### `app.state.AppState`
Stores application state. Persists when the application is reloaded.

#### API

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

#### API

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


## Usage
This GUI is designed to make it easy to segment vocalizations across multiple channels from multiple experimental sources.

### Data selection
An audio period can be selected by clicking and dragging on the spectrogram window. When Amp Env view is on, you drag a threshold line across the amplitude envelope shown. When Amp Env view is off, you select a region of time-frequency space. This region is used to detect individual vocal intervals. Clicking again (not dragging) on the spectrogram will clear the current selection.

### Keyboard Shortcuts
The keyboard shortcuts in this are designed to have your left hand on the keyboard:
```
Q W E
 A S D
  Z X            M
```
and your right hand on the mouse.

`A` and `D`: Move earlier and later in time

`W` and `S`: Automatically segment into calls (both do the same)

`Q`: Merge selected segments

`E`: Toggle between amplitude envelope mode on/off

`Z`: Automatically segment into calls (better in broadband noise)

`X`: Delete calls in selected time range

`Shift+W`, `SHIFT+S`: Increase and decrease (respectively) the amplitude envelope threshold value used in this section

`M`: Toggle between automatic dragging mode (click and drag selection automatically triggers call segmentation when released)

### Vocalization detection
TBD

### Data export
Data is exported into a pandas DataFrame saved in a pickle file. The data has columns "source_name", "source_channel", "t_start", and "t_stop" for all labeled intervals.

## API / Extensions
TBD
