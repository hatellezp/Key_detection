import os
import argparse

def clean(depth, prefix=None):

    # to remove data in fast training set or normal training se
    if prefix is None:
        prefix = ''

    # remove generated data
    print("cleaning generated data")
    os.system("rm -r data/{}keys_with_background".format(prefix))
    os.system("rm data/{}annotations.csv".format(prefix))

    if depth >= 1:
        print("cleaning unzipped data")
        os.system("rm -r data/bckgrnd")
        os.system("rm -r data/key_wb")
        os.system("rm -r data/fast")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('--depth', type=int,
                        help='depth of cleaning', default=1)

    depth = parser.parse_args().depth
    # print("depth: {}".format(depth))

    if depth == 3:
        good_answer = False
        value = ""

        while not good_answer:
            value = input(("You gave depth 3. Are you sure you want to clean "
                           "everyting? Even your result model? [N]o/[Y]es  "))

            if value.strip() in ['N', 'Y']:
                good_answer = True

        if value == 'Y':
            print("cleaning everything, hope you are sure about this")
            clean(depth)
    else:
        clean(depth)
