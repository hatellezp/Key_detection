# TODO: put more images of keys in the same file
#     I found how to generate several boxes
#     Generate your own annotation file and class names file.
#     One row for one image;
#     Row format: image_file_path box1 box2 ... boxN;
#     Box format: x_min,y_min,x_max,y_max,class_id (no space).

# for the moment the code is OS dependent

import os
import numpy as np
import data_builder.keys_with_background as kwb
from keras.preprocessing import image
from models.model_creation import load_settings


settings = load_settings()

PATH_TO_KEYS = settings["path_to_keys"]
PATH_TO_BACKGROUND = settings["path_to_background"]
PATH_TO_OUTPUT = settings["path_to_output"]
NUM_IMAGES = settings["num_images"]
KEY_SIZE_RANGE = (settings["key_size_range_low"],
                  settings["key_size_range_high"])
BACK_SIZE = settings["back_size"]


# some helper functions

# there are some .git and .idea directories that we would like to ignore
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def random_cfg(background_paths, key_paths, key_size_range, back_size):
    back_path = np.random.choice(background_paths)
    key_path = np.random.choice(key_paths)
    key_size = np.random.uniform(*key_size_range)
    x = int(np.random.uniform(back_size - key_size - 1))
    y = int(np.random.uniform(back_size - key_size - 1))
    angle = int(np.random.uniform(0, 360))
    flip = np.random.choice((True, False))
    flip_bckd = np.random.choice((True, False))
    blurr = np.random.uniform()

    return (back_path, key_path, key_size, x, y, angle, flip, flip_bckd, blurr)

#===============================================================================
#-----------------------SETUP BEGGINS HERE--------------------------------------

first_str = """Default values:
    PATH_TO_KEYS: {},
    PATH_TO_BACKGROUND: {},
    PATH_TO_OUTPUT=: {},
    NUM_IMAGES: {},
    KEY_SIZE_RANGE: {},
    BACK_SIZE: {},""".format(PATH_TO_KEYS, PATH_TO_BACKGROUND, PATH_TO_OUTPUT,
                         NUM_IMAGES, KEY_SIZE_RANGE, BACK_SIZE)

print(first_str)

# decompressing images before mixing
print("Decompressing background and keys .zip files into data/.")
try:
    os.system("unzip data/bckgrnd.zip -d data/")
    os.system("unzip data/key_wb.zip -d data/")
    print("Done decompressing.")
except Exception as e:
    print("ERROR: {}".format(e))

# create output of mixing directory if it doesn't exist
if not os.path.exists(PATH_TO_OUTPUT):
    print("Attempting to create output directory")
    try:
        os.makedirs(PATH_TO_OUTPUT)
        print("Output directory created.")
    except Exception as e:
        print("ERROR: {}".format(e))


# load paths..
print("loading keys and background image paths as lists")

# Load paths to key
key_paths = []
for path in listdir_nohidden(PATH_TO_KEYS):
    key_paths.append(os.path.join(PATH_TO_KEYS, path))

# Load paths to backgrounds
back_paths = []
for path in listdir_nohidden(PATH_TO_BACKGROUND):
    back_paths.append(os.path.join(PATH_TO_BACKGROUND, path))

csv_lines = []
num_images = NUM_IMAGES

print("generating {} mixed images".format(num_images))
# creates num_images images with keys inside as background
while num_images > 0:

    # creates a random configuration to associate background and key image
    (back_path, key_path, key_size, x, y, angle, flip, flip_bckd, blurr) = \
        random_cfg(back_paths, key_paths, KEY_SIZE_RANGE, BACK_SIZE)

    # load background and key image and make the fusion with parameters
    # from randon cfg
    b = kwb.load_background(back_path, BACK_SIZE, BACK_SIZE, flip_bckd)
    k = kwb.load_key(key_path, key_size, angle, flip)
    final = kwb.addkey_to_background(b, k, x, y, blurr)


    # Save image

    output_path = os.path.join(PATH_TO_OUTPUT, 'gen_{:06d}.jpg'.format(num_images))
    img = image.array_to_img(final)
    img.save(output_path)

    # Keep track of image bounding box

    (height, width) = k.shape[:2]
    csv_lines.append('{} {},{},{},{},0\n'.format(output_path, x,
                     y, x + width, y + height))

    # plt.imshow (final)
    # plt.show ()

    num_images -= 1
    if num_images % 100 == 0:
        print (num_images, ' left') # for python2 compatibility?

with open(os.path.join(PATH_TO_OUTPUT, 'annotations.csv'), 'w') as f:
    for l in csv_lines:
        f.write(l)


print("Moving annotations.csv to data/")
try:
    os.system("mv {} data/".format(os.path.join(PATH_TO_OUTPUT,
                                                'annotations.csv')))
except Exception as e:
    print("ERROR: {}".format(e))

print("Done, successfully generated {} mixed images in {}".format(NUM_IMAGES,
                                                                PATH_TO_OUTPUT))



