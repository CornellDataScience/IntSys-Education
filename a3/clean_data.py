import pickle
import numpy as np
from PIL import Image


def load_pickle_file(path_to_file):
  """
  Loads the data from a pickle file and returns that object
  """
  ## Look up: https://docs.python.org/3/library/pickle.html 
  ## The code should look something like this:
  # with open(path_to_file, 'rb') as f:
  #   obj = pickle.... 
  ## We will let you figure out which pickle operation to use
  pass


## You should define functions to resize, rotate and crop images
## below. You can perform these operations either on numpy arrays
## or on PIL images (read docs: https://pillow.readthedocs.io/en/stable/reference/Image.html)


## We want you to clean the data, and then create a train and val folder inside
## the data folder (so your data folder in a3/ should look like: )
# data/
#   data.zip
#   train/
#   val/

## Inside the train and val folders, you will have to dump the CLEANED images and
## labels. You can dump images/annotations in a pickle file (because our data loader 
## expects the path to a pickle file.)

## Most code written in this file will be DIY. It's important that you get to practice
## cleaning datasets and visualising them, so we purposely won't give you too much starter
## code. It'll be up to you to look up documentation and understand different Python modules.
## That being said, the task shouldn't be too hard, so we won't send you down any rabbit hole.

if __name__ == "__main__":
  ## Running this script should read the input images.pkl and labels.pkl and clean the data
  ## and store cleaned data into the data/train and data/val folders
  
  ## To correct rotated images and add missing labels, you might want to prompt the terminal
  ## for input, so that you can input the angle and the missing label
  ## Remember, the first 60 images are rotated, and might contain missing labels.
  pass
