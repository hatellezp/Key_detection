import os
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, \
    EarlyStopping

import clean
from models import model_creation as mc
from models.model_creation import models as MODELS
from setup import decompress_data, need_to_decompress, generates_examples, \
    generated_data_exists

#===============================================================================
# fast train function
def fast_train(model_name,
               model_result_h5,
               anchors,
               num_classes,
               log_dir,
               annotation_path,
               batch_size,
               epoch,
               initial_epoch,
               model_results,
               input_shape
):
    if os.path.exists(model_result_h5):
        already_trained_data = True
    else:
        raise Exception("create and train the model first")

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

        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        # train
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change

        print('Unfreeze all of the layers.')
        print('Train on {} samples, val on {} samples, with batch size {}.'
              .format(num_train, num_val, batch_size))
        model.fit_generator(mc.data_generator_wrapper(
            lines[:num_train],
            batch_size,
            input_shape,
            anchors,
            num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=
            mc.data_generator_wrapper(lines[num_train:],
                                      batch_size,
                                      input_shape,
                                      anchors,
                                      num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch,
            initial_epoch=initial_epoch,
            callbacks=[logging, checkpoint, reduce_lr,
                       early_stopping]
        )
        model.save_weights(model_results + model_name + '_weights.h5')

if __name__ == '__main__':


    # settings are stored in 'default_settings.json' or 'settings.json' if it exists
    settings = mc.load_fast_settings()

    INITIAL_EPOCH = settings["initial_epoch"]
    EPOCH = settings["epoch"]
    BATCH_SIZE = settings["batch_size"]
    PASSES = settings["passes"]
    NUM_IMAGES = settings["num_images"]

    ANNOTATION_PATH = settings["annotation"]

    LOG_DIR = settings["logs"]
    if not os.path.exists(LOG_DIR):  # verify if the directory exists
        os.makedirs(LOG_DIR)

    MODEL_RESULTS = "model_data/"
    if not os.path.exists(MODEL_RESULTS):  # verify if the directory exists
        os.makedirs(MODEL_RESULTS)

    CLASSES_PATH = settings["classes"]
    ANCHORS_PATH = settings["anchors"]

    CLASS_NAMES = mc.get_classes(CLASSES_PATH)
    NUM_CLASSES = len(CLASS_NAMES)

    ANCHORS = mc.get_anchors(ANCHORS_PATH)
    INPUT_SHAPE = (416, 416)  # multiple of 32, hw

    MODEL_NAME = settings["model_name"]
    WEIGHT_ZERO_PATH = settings["weights"]

    MODEL_RESULT_H5 = MODEL_RESULTS + MODEL_NAME + "_weights.h5"

    PATH_TO_KEYS = settings["path_to_keys"]
    PATH_TO_BACKGROUND = settings["path_to_background"]
    PATH_TO_OUTPUT = settings["path_to_output"]
    NUMBER_KEYS = settings["number_keys"]
    KEY_SIZE_RANGE = (settings["key_size_range_low"],
                      settings["key_size_range_high"])
    BACK_SIZE = settings["back_size"]
    CROP_BOUND = settings["crop"]
    ROOT = settings["data_root"]

    # several passes
    print("Doing {} passes".format(PASSES))
    for i in range(PASSES):

        # clean the fast training set before you
        clean.clean(0, prefix="fast/")

        generates_examples(PATH_TO_KEYS,
                           PATH_TO_BACKGROUND,
                           PATH_TO_OUTPUT,
                           NUM_IMAGES,
                           NUMBER_KEYS,
                           KEY_SIZE_RANGE,
                           BACK_SIZE,
                           CROP_BOUND,
                           ROOT)

        # fast train
        fast_train(
            MODEL_NAME,
            MODEL_RESULT_H5,
            ANCHORS,
            NUM_CLASSES,
            LOG_DIR,
            ANNOTATION_PATH,
            BATCH_SIZE,
            EPOCH,
            INITIAL_EPOCH,
            MODEL_RESULTS,
            INPUT_SHAPE
        )

        # clean after you
        clean.clean(0, prefix='fast/')

    # clean fast directory
    clean.clean(1)

