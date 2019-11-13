import os
import tensorflow as tf
import sys
import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, \
    EarlyStopping
from models.yolo3.model import preprocess_true_boxes, yolo_body, \
    tiny_yolo_body, yolo_loss
from models.yolo3.utils import get_random_data




# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--annotation', type=str, default='annotations.csv',
    help='path to model weight file, default '
)
parser.add_argument(
    '--classes', type=str, default='model_data/key_classes.txt',
    help='path to model weight file, default '
)
parser.add_argument(
    '--anchors', type=str, default='model_data/yolo_anchors.txt',
    help='path to model weight file, default '
)
parser.add_argument(
    '--initial_epoch1', type=int, default=0,
    help='initial_epoch1 with frozen layers first, to get a stable loss. '
)
parser.add_argument(
    '--initial_epoch2', type=int, default=100,
    help='initial_epoch2 Unfreeze and continue training, to fine-tune. '
)
parser.add_argument(
    '--epoch1', type=int, default=100,
    help='epoch1 with frozen layers first, to get a stable loss.'
)
parser.add_argument(
    '--epoch2', type=int, default=200,
    help='epoch2 Unfreeze and continue training, to fine-tune. '
)
parser.add_argument(
    '--batch_size1', type=int, default=64,
    help='batch_size1 with frozen layers first, to get a stable loss. '
)
parser.add_argument(
    '--batch_size2', type=int, default=64,
    help='batch_size2 of the Unfreeze and continue training, to fine-tune. '
)
FLAGS = parser.parse_args()
annotation_path = FLAGS.annotation
log_dir = 'logs/'
model_results = "weights_yolo_train/"
if not os.path.exists(model_results):  # vérification si le dossier existe
    os.makedirs(model_results)
classes_path = FLAGS.classes
anchors_path = FLAGS.anchors
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (416, 416)  # multiple of 32, hw
is_tiny_version = len(anchors) == 6  # default setting
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
                              freeze_body=2, weights_path='model_data/yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2, weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(model_results + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1)
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines) * val_split)
num_train = len(lines) - num_val
# Train with frozen layers first, to get a stable loss. Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})
    batch_size = FLAGS.batch_size1
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                               num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=FLAGS.epoch1,
                        initial_epoch=FLAGS.initial_epoch1,
                        callbacks=[logging, checkpoint])
    model.save_weights(model_results + 'trained_weights_stage_1.h5')
