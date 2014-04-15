# WearScript EyeTracking #

See http://wearscript.com for project details and http://www.wearscript.com/en/latest/eyetracking.html for eyetracking specific documentation.

## Installation ##

Install https://github.com/wearscript/wearscript-python first.

Ubuntu (13.10) Dependencies

```bash
sudo apt-get install python-opencv python-matplotlib
```

## Usage ##

Each script has documentation about it's arguments if you run it like "python track.py" for the track.py script.

* track.py: For performing eye tracking and publishing the pupil center using PubSub in WearScript.
* track_debug.py: Uses track.py but is intended for debugging by showing you the pupil and detected regions.
* calibrate.py (UNSUPPORTED CURRENLTY): Used to generate an eye-to-world model.
* combine.py (UNSUPPORTED CURRENLTY): Used to visualize the eye-to-world model with data saved with track.py.
