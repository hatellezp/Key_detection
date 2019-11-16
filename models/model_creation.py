import os
import json
import csv
import numpy as np

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from models.yolo3.model import preprocess_true_boxes, yolo_body, \
    tiny_yolo_body, yolo_loss
from models.yolo3.utils import get_random_data
################################################################################
################################################################################

def load_settings():
    # if new settings exists use them, else load default
    if os.path.exists("settings.json"):
        json_path = "settings.json"
    else:
        json_path = "default_settings.json"

    res = {}
    with open(json_path) as json_file:
        data = json.load(json_file)

    # we do this to ensure that all needed values are present and that only
    res.update(data)
    return res

def load_fast_settings():
    # if new settings exists use them, else load default
    if os.path.exists("fast_settings.json"):
        json_path = "fast_settings.json"
    else:
        json_path = "default_fast_settings.json"

    res = {}
    with open(json_path) as json_file:
        data = json.load(json_file)

    # we do this to ensure that all needed values are present and that only
    res.update(data)
    return res


# each time reads names.csv to see what models are valid
# load valid models names
def load_valid_model_names():
    valid_model_names = []

    with open("models/names.csv") as f:
        reader = csv.reader(f, delimiter=',', )
        for row in reader:
            if not ('names' in row[0].strip()):
                valid_model_names.append(row[0].strip())

    return valid_model_names


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


################################################################################
################################################################################
# models creation

def create_model_yolov3(input_shape, anchors, num_classes, load_pretrained=True,
                 freeze_body=2, weights_path='model_data/yolo_weights.h5'):

    '''create the training model'''
    K.clear_session() # get a new session

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],
                           num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)

    print('Create YOLOv3 model with {} anchors and {} classes.'
          .format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False

            print('Freeze the first {} layers of total {} layers.'
                  .format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes,
                   'ignore_thresh': 0.5}
                        )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)
    return model


def create_model_yolov3_tiny(input_shape, anchors, num_classes,
                             load_pretrained=True,
                             freeze_body=2,
                             weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'
          .format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'
                  .format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.7}
                        )(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)
    return model


def create_model_darknet53(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    pass


models = {
    "yolov3": create_model_yolov3,
    "yolov3-tiny": create_model_yolov3_tiny,
    "darknet53": create_model_darknet53,
}


################################################################################
################################################################################

# data generators

def data_generator(annotation_lines, batch_size, input_shape, anchors,
                   num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0

    while True:
        image_data = []
        box_data = []

        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape,
                                         random=True)

            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n

        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors,
                                       num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors,
                           num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None

    return data_generator(annotation_lines, batch_size, input_shape, anchors,
                          num_classes)




# for test
if __name__ == "__main__":
    print(load_valid_model_names())
    print(load_settings())

