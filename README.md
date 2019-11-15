# Key_detection

Key_detection is a library that (as the names says) detects keys
in images. It proposes the uses of several deep neural networks structures.
It can also be used to detect keys in video files.

This work is greatly inspired from [belarbi2733/keras_yolov3](https://github.com/belarbi2733/keras_yolov3). 

## Installation

### Requirements:
    keras_preprocessing==1.0.5
    keras_applications==1.0.7
    keras==2.2.0
    keras_applications==1.0.7
    
There some problems using _tensorflow==2.*_ so don't use it.
If you have a different setup, please let me know.

## Setup

### Weights:
Pretrained weights for yolov3-tiny are already present. You can download
more from [here](https://pjreddie.com/darknet/).

### Strucure
The project is organized as follows:
```bash
.
├── data
│   ├── bckgrnd.zip
│   └── key_wb.zip
├── data_builder
│   └── keys_with_background.py
├── data_cfg
│   ├── darknet53.cfg
│   ├── yolov3.cfg
│   └── yolov3-tiny.cfg
├── font
│   ├── FiraMono-Medium.otf
│   └── SIL Open Font License.txt
├── model_data
│   ├── coco_classes.txt
│   ├── key_classes.txt
│   ├── tiny_yolo_anchors.txt
│   ├── voc_classes.txt
│   ├── yolo_anchors.txt
│   └── yolov3-tiny_zero.weights
├── models
│   ├── yolo3
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── convert.py
│   ├── kmeans.py
│   ├── model_creation.py
│   ├── names.csv
│   └── yolo.py
├── clean.py
├── default_settings.json
├── detect_key.py
├── fast_train.py
├── README.md
├── setup.py
└── train.py
```

The main files are:
* _default_settings.json_
* _setup.py_
* _train.py_
* _fast_train.py_
* _detect_key.py_
* _clean.py_

### Default setup:
Default settings are provided in _default_settings.json_. If you want to change 
anything:
* change model loaded
* provide a different training set
* provide a different validation set

then create your own settings file _settings.json_, don't overwrite defaults **(please)**.

parameters in _default_settings.json_:
* model_name: a name of an implemented model, valid models are in 'models/names.csv'
* annotation: contains the trained data
* classes: classes you which to train the model to
* anchors: contains how many anchors and representative class value for each class
* weigths: some pretrained weights to not do the work from the beginning
* configuration: .cfg file to build your specified model
* logs: directory to save logs if you want to
* for the freezed phase:
	* initial_epoch1: the name describes it
	* epoch1: same
	* batch_size1: guess it
* for the unfreezed phase you have the same parameters with number 2 as suffix
* training: defines type of training done with _train.py_:
	* 1: only freezed
	* 2: only unfreezed
	* 3: whole
* path_to_keys: name says it all
* path_to_background : ...
* now same parameters as training phases with 'fast_' as prefix for the _fast_train.py_ file

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

Clean after (or before) you. If you want to delete the generated training data and start over
(or clean even more: modify **DEFAULT** in _clean.py_)
```bash
python clean.py
```

## Contributions
Do what you want, is a free world out there.

