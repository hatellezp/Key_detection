import os
import numpy as np

import data_builder.keys_with_background as kwb
import clean

from keras.preprocessing import image
from models.model_creation import load_settings

################################################################################
################################################################################
settings = load_settings()

PATH_TO_KEYS = settings["path_to_keys"]
PATH_TO_BACKGROUND = settings["path_to_background"]
PATH_TO_OUTPUT = settings["path_to_output"]
NUM_IMAGES = settings["num_images"]
KEY_SIZE_RANGE = (settings["key_size_range_low"],
                  settings["key_size_range_high"])
BACK_SIZE = settings["back_size"]
CROP_BOUND = settings["crop"]
NUMBER_KEYS = settings["number_keys"]
ROOT = settings["data_root"]

# some helper functions

# there are some .git and .idea directories that we would like to ignore
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def random_cfg(background_paths, key_paths, key_size_range, back_size,
               crop_bound):
    back_path = np.random.choice(background_paths)
    key_path = np.random.choice(key_paths)
    key_size = np.random.uniform(*key_size_range)
    x = int(np.random.uniform(back_size - key_size - 1))
    y = int(np.random.uniform(back_size - key_size - 1))
    angle = int(np.random.uniform(0, 360))
    flip = np.random.choice((True, False))
    flip_bckd = np.random.choice((True, False))
    blurr = np.random.uniform()

    # crop a percentage of the image
    # garuanteed we keep som 'reasonable bounds'
    if crop_bound != 0:
        crop_bound = max(min(crop_bound, 0.5), 0.05)
        crop = np.random.uniform(0.03, crop_bound)
    else:
        crop = 0

    side = np.random.choice(6)

    return (back_path, key_path, key_size, x, y, angle, flip, flip_bckd, blurr,
            crop, side)

#===============================================================================
# this big function generates the examples

def need_to_decompress():
    bckgrnd_exists = os.path.exists("data/bckgrnd")
    key_wb_exists = os.path.exists("data/key_wb")

    return not(bckgrnd_exists and key_wb_exists)

def decompress_data():
    print("Decompressing background and keys .zip files into data/.")
    try:
        os.system("unzip data/bckgrnd.zip -d data/")
        os.system("unzip data/key_wb.zip -d data/")
        print("Done decompressing.")
    except Exception as e:
        print("ERROR: {}".format(e))

def generated_data_exists(path):
    return os.path.exists(path)

def generates_examples(
        path_to_keys,
        path_to_background,
        path_to_output,
        number_of_images,
        number_keys,
        key_size_range,
        back_size,
        crop_bound,
        root,
):

    settings_used = """
        path_to_keys: {},
        path_to_background: {},
        path_to_output: {},
        number_of_images: {},
        number_keys: {},
        key_size_range: {},
        back_size: {},
        crop_bound: {}, 
    """.format(path_to_keys, path_to_background, path_to_output,
               number_of_images, number_keys, key_size_range,
               back_size, crop_bound)

    print("using settings:")
    print(settings_used)

    # decompressing images before mixing
    if need_to_decompress():
        print("need to decompress the .zip files")
        decompress_data()

    # if generated data exists, remove it
    if generated_data_exists(path_to_output):
        print("old generated data exists, removing it")
        clean.clean(0)

    # create output of mixing directory if it doesn't exist
    if not os.path.exists(path_to_output):
        print("Attempting to create output directory")
        try:
            os.makedirs(path_to_output)
            print("Output directory created.")
        except Exception as e:
            print("ERROR: {}".format(e))

    # load paths..
    print("loading keys and background image paths as lists")

    # Load paths to key
    key_paths = []
    for path in listdir_nohidden(path_to_keys):
        key_paths.append(os.path.join(path_to_keys, path))

    # Load paths to backgrounds
    back_paths = []
    for path in listdir_nohidden(path_to_background):
        back_paths.append(os.path.join(path_to_background, path))

    csv_lines = []
    num_images = number_of_images

    print("generating {} mixed images".format(num_images))
    # creates num_images images with keys inside as background
    while num_images > 0:

        # the generator of examples put now several keys in a same background
        # a random configuration is made and used to put the keys in the background

        # how many keys to be added
        key_number_list = [1]
        for i in range(2, (number_keys)):
            key_number_list.append(i)
        key_number = np.random.choice(np.array(key_number_list))

        # initial values before generating the list of image of keys to be added
        line_suffix_of_csv = ""
        keys_to_background = []
        back_path = ""
        xs = []
        ys = []
        blurrs = []
        flip_bckd = ""

        for i in range(key_number):
            # creates a random configuration to associate background and key image
            # takes as argument the list of paths to keys and backgrounds,
            # the size of the keys, the background size and how much the image of
            # the key is to be cropped
            keys_values = random_cfg(back_paths, key_paths, key_size_range,
                                     back_size, crop_bound)

            # extract the necessary values from 'keys_values' (it is a tuple)
            x = keys_values[3]
            y = keys_values[4]
            xs.append(x)
            ys.append(y)
            blurrs.append(keys_values[8])

            # load a key following the configuration
            k = kwb.load_key2(
                keys_values[1],
                keys_values[2],
                keys_values[5],
                keys_values[6],
                keys_values[9],
                keys_values[10]
            )

            # append the key to the list of keys
            keys_to_background.append(k)

            # define the back path
            back_path = keys_values[0]
            flip_bckd = keys_values[7]

            # add the box to the line in the csv file (annotations.csv) which is
            # the training set
            (height, width) = k.shape[:2]
            line_suffix_of_csv += '{},{},{},{},0 '.format(x, y, x + width,
                                                          y + height)

        # load background and key image and make the fusion with parameters
        # from randon cfg
        b = kwb.load_background(back_path, back_size, back_size, flip_bckd)
        final = kwb.addkey_to_background2(b, keys_to_background, xs, ys, blurrs[0])

        # Save image
        output_path = os.path.join(path_to_output, 'gen_{:06d}.jpg'
                                   .format(num_images))
        img = image.array_to_img(final)
        img.save(output_path)

        # creates the line which is the training example adding the prefix
        # and after the several boxes
        line_suffix_of_csv = line_suffix_of_csv[:-1]
        line_of_csv = output_path + ' ' + line_suffix_of_csv + '\n'

        # list_of_h_w = [k.shape[:2] for k in keys_to_background]
        # add line to the csv_lines
        csv_lines.append(line_of_csv)

        num_images -= 1
        if num_images % 100 == 0:
            print(num_images, ' left')  # for python2 compatibility?

    # write the examples to the .csv
    with open(os.path.join(path_to_output, 'annotations.csv'), 'w') as f:
        for l in csv_lines:
            f.write(l)

    print("Moving annotations.csv to {}".format(root))
    try:
        os.system("mv {} {}".format(
            os.path.join(path_to_output, 'annotations.csv'), root))
    except Exception as e:
        print("ERROR: {}".format(e))

    print("Done, successfully generated {} mixed images in {}"
          .format(number_of_images, path_to_output))


#===============================================================================
#-----------------------SETUP BEGGINS HERE--------------------------------------


if __name__ == "__main__":
    generates_examples(PATH_TO_KEYS,
                       PATH_TO_BACKGROUND,
                       PATH_TO_OUTPUT,
                       NUM_IMAGES,
                       NUMBER_KEYS,
                       KEY_SIZE_RANGE,
                       BACK_SIZE,
                       CROP_BOUND,
                       ROOT)

