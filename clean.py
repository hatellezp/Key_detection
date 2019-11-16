import  os

DEFAULT = 1 # 1: leave model alone, 2: remove model to

def clean(depth):
    # remove generated data
    print("cleaning generated data")
    os.system("rm -r data/keys_with_background")
    os.system("rm data/annotations.csv")

    if depth >= 1:
        print("cleaning unzipped data")
        os.system("rm -r data/bckgrnd")
        os.system("rm -r data/key_wb")

    # remove unnecessary files result of training
    if depth >= 2:
        print("cleaning temporary models")
        os.system("rm model_data/ep*.h5")
        os.system("rm model_data/*.png")

    # remove trained model
    # BE SURE YOU WANT THIS
    if depth >= 3:
        print("cleaning weights results")
        os.system("rm model_data/*.h5")


if __name__ == "__depth__":
    clean(1)
