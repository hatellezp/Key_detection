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
    # call on convert._main to create model
    config_path = settings["configuration"] + model_name + ".cfg"
    # weights_path = weights_zero_path
    convert._main(config_path, weights_zero_path, model_result_h5)
    already_trained = True
    print("done, .h5 configuration created")
