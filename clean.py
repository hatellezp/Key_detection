import  os

DEFAULT = 1 # 1: leave model alone, 2: remove model to

# remove generated data
os.system("rm -r data/bckgrnd")
os.system("rm -r data/key_wb")
os.system("rm -r data/keys_with_background")
os.system("rm data/annotations.csv")

# remove unnecessary files result of training
os.system("rm model_data/ep*.h5")
os.system("rm model_data/*.png")

# remove trained model
# BE SURE YOU WANT THIS
if DEFAULT == 2:
    os.system("rm model_data/*.h5")
