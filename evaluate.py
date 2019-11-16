import numpy as np
import os
import argparse

from models.model_creation import models as MODELS
import models.model_creation as mc
import setup
import clean

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, \
    EarlyStopping


def evaluate_model(
        model_name,
        model_result_h5,
        anchors,
        input_shape,
        num_classes,
        log_dir,
        model_results,
        annotation_path,
        batch_size
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

        print("all layers unfreezed")
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        # train
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred},
                      metrics=['mean_squared_error'])  # recompile to apply the change


        steps = int(len(lines) / batch_size)

        result_of_evaluation = \
            model.evaluate_generator(
                mc.evaluation_generator_w(lines,
                                            batch_size,
                                            input_shape,
                                            anchors,
                                            num_classes),
                steps=steps,
                verbose=1
                        )

        return result_of_evaluation



if __name__ == "__main__":
    # create evaluation function here
    settings = mc.load_settings()

    # load settings particular to evaluate.py
    evaljson = {}
    with open("evaluate.json") as json_file:
        data = json.load(json_file)
    evaljson.update(data)

    #=========================================================================
    # note that sometimes the values are loaded from evaluate.json
    NUMBER_OF_EVALUATIONS = evaljson["number_of_evaluations"]
    NUM_IMAGES = evaljson["num_images"]  # here for example
    BATCH_SIZE = evaljson["batch_size"]

    PATH_TO_KEYS = settings["path_to_keys"]
    PATH_TO_BACKGROUND = settings["path_to_background"]
    PATH_TO_OUTPUT = "data/evaluation/keys_with_background"
    NUMBER_KEYS = settings["number_keys"]
    KEY_SIZE_RANGE = (settings["key_size_range_low"],
                      settings["key_size_range_high"])
    BACK_SIZE = settings["back_size"]
    CROP_BOUND = settings["crop"]
    ROOT = "data/evaluation"
    MODEL_NAME = settings["model_name"]
    MODEL_RESULTS = "model_data/"
    if not os.path.exists(MODEL_RESULTS):  # verify if the directory exists
        os.makedirs(MODEL_RESULTS)

    MODEL_RESULT_H5 = MODEL_RESULTS + MODEL_NAME + "_weights.h5"
    INITIAL_EPOCH = settings["initial_epoch1"]
    EPOCH = settings["epoch1"]

    ANNOTATION_PATH = "data/evaluation/annotations.csv"

    LOG_DIR = settings["logs"]
    if not os.path.exists(LOG_DIR):  # verify if the directory exists
        os.makedirs(LOG_DIR)


    CLASSES_PATH = settings["classes"]
    ANCHORS_PATH = settings["anchors"]

    CLASS_NAMES = mc.get_classes(CLASSES_PATH)
    NUM_CLASSES = len(CLASS_NAMES)

    ANCHORS = mc.get_anchors(ANCHORS_PATH)
    INPUT_SHAPE = (416, 416)  # multiple of 32, hw

    WEIGHT_ZERO_PATH = settings["weights"]

    MODEL_RESULT_H5 = MODEL_RESULTS + MODEL_NAME + "_weights.h5"

    #########################################################################
    # parser here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('--output', type=str,
                        help='depth of cleaning', default=None)

    output = parser.parse_args().output

    ########################################################################

    # we are going to generate n number of evaluations
    # a compute the mean of all results
    # is a good idea to have a low number of images generated
    number_of_evaluations = 10
    results = []
    mean_loss = 0
    mean_mean_squared_error = 0


    for i in range(NUMBER_OF_EVALUATIONS):

        ""
        clean.clean(0, prefix="evaluation/")

        setup.generates_examples(PATH_TO_KEYS,
                                 PATH_TO_BACKGROUND,
                                 PATH_TO_OUTPUT,
                                 NUM_IMAGES,
                                 NUMBER_KEYS,
                                 KEY_SIZE_RANGE,
                                 BACK_SIZE,
                                 CROP_BOUND,
                                 ROOT)

        result = evaluate_model(
                MODEL_NAME,
                MODEL_RESULT_H5,
                ANCHORS,
                INPUT_SHAPE,
                NUM_CLASSES,
                LOG_DIR,
                MODEL_RESULTS,
                ANNOTATION_PATH,
                BATCH_SIZE
        )

        (r_loss, r_mean) = (result[0], result[1])

        # compute the resulting means
        mean_loss = (mean_loss*i + r_loss) / float(i+1)
        mean_mean_squared_error = (mean_mean_squared_error * i + r_mean) / float(i + 1)

        results.append((r_loss, r_mean))

        print("at step {}:\n    mean loss: {}\n    mean mse: {}"
              .format(i+1, mean_loss, mean_mean_squared_error))

        clean.clean(0, prefix="evaluation/")

    clean.clean(1)

    with open(output, 'w') as f:

        string_output = """
=====================================================================
    evaluating model: {}
    batch_size: {}
    number of steps: {}
    number of examples by step: {} 
======================================================================
         
        """.format(MODEL_NAME, BATCH_SIZE, number_of_evaluations, NUM_IMAGES)
        f.write(string_output)

        for i in range(len(results)):

            string_output =  "- at step {}:\n    loss: {}\n    mse: {}\n"\
                .format(i + 1, results[i][0], results[i][1])

            print(string_output)
            f.write(string_output)

        string_output = "= final result:\n    mean loss: {}\n    mean mse: {}\n"\
            .format(mean_loss, mean_mean_squared_error)

        print(string_output)
        f.write(string_output)
