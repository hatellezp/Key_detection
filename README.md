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
    opencv
    
There some problems using _tensorflow==2.*_ so don't use it.
If you have a different _working_ setup, please let me know.


### Weights:
Pretrained weights for yolov3-tiny are already present. You can download
more from [here](https://pjreddie.com/darknet/).


### Structure
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
├── default_fast_settings.json
├── default_settings.json
├── detect_key.py
├── fast_train.py
├── README.md
├── setup.py
└── train.py
```

The main files are (in order of use):
* _default_settings.json_
* _default_fast_settings.json_
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

then create your own settings file _settings.json_, don't overwrite defaults **(please)** 
(_this is also true for default_fast_settings.json_).


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


### Setup
For training with the supplied data first call _setup.py_:
```bash
python setup.py
```
This file will generate examples in the form of a .csv file and a directory of
images.
Each line in the .csv file follows the syntax:

    example : path_to_image box_1,class_1 box_2,class_2, ... box_N,class_N
    box_i   : x_coordinate,y_coordinate,widht,height 
    
note that are spaces between each pair _box,class_ and no space in the 
specification of a box.

What *settings.json parameters are important for _setup.py_:
* annotation: path to the .csv file
* path_to_keys: path to the key-only images
* path_to_background: path to the back background-only images
* path_to_output: path to where the mixed images will live
* num_images: number of examples to generate
* key_size_range_low: lower bound of the size of the result keys
* key_size_range_high: upper bound of the size of the result keys
* back_size: size of the background image
* crop: keys will be cropped before mix with a percentange between 
    (5%, (100*crop)%)
* number_keys: there will be between (1, number_keys) in each result image 


### Train
After _what I would do is train the model..._

Train the model
```bash
python train.py 
```

What *settings.json parameters are important for _setup.py_:
* model_name: yolov3, yolov3-tiny... model to be used
* classes: path to file with the target number of classes
* anchors: path to anchors file
* weights: path to *.weights file with pretrained values
* configuration: path to *.cfg file with the configuration of the
  network
* logs: path to logs dir to keep track if needed
* initial_epoch(1,2): value for initial epoch
* epoch(1,2): number of epochs
* batch_size(1,2): batch_size
* training: type of training,
    * 1: only train with freezed layers
    * 2: train with unfreezed layers
    * 3: perform 1-training and then 2-training

The numbers (1,2) in the parameters are for which type of training you want to
adjust.    


### Evaluation
Use the command
```bash
python evaluate.py -output=
```

### Key detection
This part is really simple. You will be using the parameters in *settings.json.

Key detection in a image 
```bash
python detect_key.py  --image --input <path-to-image> --output <path-to-result-image>
```

Key detection in a video
```bash
python detect_key.py  --video --input <path-to-video> --output <path-to-result-video>
```

### Adjust
You can always perform a "fast" training that generates new data, train your
network in it and clean after. Parameters are the same as *settings.json, but
this time they live in *fast_settings.json.
```bash
python fast_train.py 
```


### Clean
Clean after (or before) you. You can call it anytime. You have different depths:
* 0: clean only generated data
* 1: clean unzipped data
* 2: clean temporary *.h5
* 3: clean the result model *.h5

The depth value is increasing, calling with 2 will perform 0,1 and 2.
Default is 1.

```bash
python clean.py --depth=d
```

## Contributions
Do what you want, is a free world out there.

