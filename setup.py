# for the moment the code is OS dependet

import os

PATH_TO_KEYS = "data/key_wb"
PATH_TO_BACKGROUND = "data/bckgrnd"
PATH_TO_OUTPUT = "data/keys_with_background"


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




