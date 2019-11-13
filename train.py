import os
import tensorflow as tf
import sys
import argparse
import numpy as np


# import keras.backend as K
import tensorflow.keras.backend as K
# from keras.layers import Input, Lambda
# from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from models.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
# from models.yolo3.utils import get_random_data

from models import model_creation as mc
from models.model_creation import models as MODELS
from models import  convert

# settings are stored in 'default_settings.json' or 'settings.json' if it exists
settings = mc.load_settings()

# batch_size, epoch and initial_epochs for both phases
initial_epoch1 = settings["initial_epoch1"]
initial_epoch2 = settings["initial_epoch2"]
epoch1 = settings["epoch1"]
epoch2 = settings["epoch2"]
batch_size1 = settings["batch_size1"]
batch_size2 = settings["batch_size2"]


annotation_path = settings["annotation"]

log_dir = settings["logs"]
if not os.path.exists(log_dir):  # verify if the directory exists
    os.makedirs(log_dir)

model_results = "model_data/"
if not os.path.exists(model_results):  # verify if the directory exists
    os.makedirs(model_results)

classes_path = settings["classes"]
anchors_path = settings["anchors"]

class_names = mc.get_classes(classes_path)
num_classes = len(class_names)

anchors = mc.get_anchors(anchors_path)
input_shape = (416, 416)  # multiple of 32, hw


model_name = settings["model_name"]
weights_zero_path = settings["weights"]

model_result_h5 = model_results + model_name + "_weights.h5"

# if the model was trained at least once, load the weights
# if not call convert.py
if os.path.exists(model_result_h5):
    already_trained_data = True
else:
    # call on convert._main to create model
    config_path = settings["configuration"] + model_name + ".cfg"
    # weights_path = weights_zero_path
    convert._main(config_path, weights_zero_path, model_result_h5)
    already_trained = True

# verify the model has been implemented
if model_name in mc.load_valid_model_names():
    create_model = MODELS[model_name]

    if model_name == "yolov3-tiny" and len(anchors) != 6:
        raise Exception("ERROR: incorrent number of anchors for yolov3-tiny")

    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2,
                         weights_path=model_result_h5)


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


    # here we get what type of training is done:
    # 1 : only freezed
    # 2:  only unfreezed
    # 3: whole
    training = settings["training"]

    # Train with frozen layers first, to get a stable loss. Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if training in [1, 3]:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'
              .format(num_train, num_val, batch_size1))
        model.fit_generator(mc.data_generator_wrapper(
            lines[:num_train],
            batch_size1,
            input_shape,
            anchors,
            num_classes),
                            steps_per_epoch=max(1, num_train // batch_size1),
                            validation_data=
                            mc.data_generator_wrapper(lines[num_train:],
                                                      batch_size1,
                                                      input_shape,
                                                      anchors,
                                                      num_classes),
                            validation_steps=max(1, num_val // batch_size1),
                            epochs=epoch1,
                            initial_epoch=initial_epoch1,
                            callbacks=[logging, checkpoint]
        )

        if training == 1:
            model.save_weights(model_results + model_name + '_weights.h5')

    # second phase
    if training in [2, 3]:
        # unfreeze layers
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        # train
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change

        print('Unfreeze all of the layers.')
        print('Train on {} samples, val on {} samples, with batch size {}.'
              .format(num_train, num_val, batch_size2))
        model.fit_generator(mc.data_generator_wrapper(
            lines[:num_train],
            batch_size2,
            input_shape,
            anchors,
            num_classes),
                            steps_per_epoch=max(1, num_train // batch_size2),
                            validation_data=
                            mc.data_generator_wrapper(lines[num_train:],
                                                      batch_size2,
                                                      input_shape,
                                                      anchors,
                                                      num_classes),
                            validation_steps=max(1, num_val // batch_size2),
                            epochs=epoch2,
                            initial_epoch=initial_epoch2,
                            callbacks=[logging, checkpoint, reduce_lr,
                                       early_stopping]
        )
        model.save_weights(model_results + model_name + '_weights.h5')
    # Further training if needed.










