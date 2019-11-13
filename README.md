# Key_detection

Key_detection is a library that (as the names says) detects keys
in images. It proposes the uses of several deep neural networks structures.
It can also be used to detect keys in video files.

## Installation

No installation needed. Clone the repository and enjoy.

## Usage

For training with the supplied data first call _setup.py_:
```bash
python setup.py
```

Default settings are provided in _default_settings.json_. If you want to change 
anything:
* change model loaded
* provide a different training set
* provide a different validation set

then create your own settings file _settings.json_, don't overwrite defaults.

Key detection in a image 
```bash
python detect_key.py  --image --input <path-to-image> --output <path-to-result-image>
```

Key detection in a video
```bash
python detect_key.py  --video --input <path-to-video> --output <path-to-result-video>
```

Retrain model
```bash
train.py 
```


## Yes of course, nothing works yet...