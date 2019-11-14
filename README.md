# Key_detection

Key_detection is a library that (as the names says) detects keys
in images. It proposes the uses of several deep neural networks structures.
It can also be used to detect keys in video files.

## Installation

No installation needed. Clone the repository and enjoy.

## Usage

Several models can be implemented. For the moment we use only

* _yolov3_ 
* _yolov3-tiny_ 

because easiness of installation and quality of results.
For each model you should download pretrained weights from 
(https://pjreddie.com/darknet/yolo/) or **(as a crazy person train the model from
zero: use the Pscal VOC data)**.

Again, go to (https://pjreddie.com/darknet/yolo/), is a better place than this
repository, or stay here, who am I to judge...

Default settings are provided in _default_settings.json_. If you want to change 
anything:
* change model loaded
* provide a different training set
* provide a different validation set

then create your own settings file _settings.json_, don't overwrite defaults **(please)**.

For training with the supplied data first call _setup.py_:
```bash
python setup.py
```

After _what I would do is train the model..._
Train the model
```bash
python train.py 
```
or perform a fast training to adjust a little
```bash
python fast_train.py 
```

Key detection in a image 
```bash
python detect_key.py  --image --input <path-to-image> --output <path-to-result-image>
```

Key detection in a video
```bash
python detect_key.py  --video --input <path-to-video> --output <path-to-result-video>
```

## Contributions
Do what you want, is a free world out there.

