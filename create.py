import os

from models import model_creation as mc
from models import convert

#===============================================================================
# settings are stored in 'default_settings.json' or 'settings.json' if it exists
settings = mc.load_settings()

model_results = "model_data/"
if not os.path.exists(model_results):  # verify if the directory exists
    os.makedirs(model_results)

model_name = settings["model_name"]
weights_zero_path = settings["weights"]

model_result_h5 = model_results + model_name + "_weights.h5"

################################################################################

# if the model was trained at least once, load the weights
# if not call convert.py
if os.path.exists(model_result_h5):
    already_trained_data = True
    print("{} already exists, not creating one from {}"
          .format(model_result_h5,
                  (settings["configuration"] + model_name + ".cfg")))
else:
    print(("{} doestn't exists, creating from {}"
           .format(model_result_h5,
                   (settings["configuration"] + model_name + ".cfg"))))

    # if weight_zero_path don't exist, attempt to create them
    if not os.path.exists(weights_zero_path):
        print("no weights in {}".format(weights_zero_path))
        print("attempting to get them")
        if weights_zero_path in ["model_data/yolov3_zero.weights",
                                 "model_data/yolov3-tiny_zero.weights"]:
            prefix = "wget https://pjreddie.com/media/files/"
            if weights_zero_path == "model_data/yolov3_zero.weights":
                os.system(prefix + "yolov3.weights")
                os.system("mv yolov3.weights model_data/yolov3_zero.weights")
                print("weights succefully got and setup")
            elif weights_zero_path == "model_data/yolov3-tiny_zero.weights":
                os.system(prefix + "yolov3-tiny.weights")
                os.system("mv yolov3-tiny.weights model_data/yolov3-tiny_zero.weights")
                print("weights succefully got and setup")
        else:
            raise Exception("no weights in defined location and no method to get them")

    # call on convert._main to create model
    config_path = settings["configuration"] + model_name + ".cfg"
    # weights_path = weights_zero_path
    convert._main(config_path, weights_zero_path, model_result_h5)
    already_trained = True
    print("done, .h5 configuration created")
