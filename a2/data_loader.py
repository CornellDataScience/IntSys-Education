import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class SimpleDataset(Dataset):
  def __init__(self, path_to_csv):
    ## TODO: Add code to read csv and load data. 
    ## You should store the data in a field.
    # Eg (on how to read .csv files):
    # with open('path/to/.csv', 'r') as f:
    #   lines = ...
    ## Look up how to read .csv files using Python. This is common for datasets in projects.
    pass

  def __len__(self):
    ## TODO: Returns the length of the dataset.
    pass

  def __getitem__(self, index):
    ## TODO: This returns only ONE sample from the dataset, for a given index.
    ## The returned sample should be a tuple (x, y) where x is your input 
    ## vector and y is your label
    ## Remember to convert the x and y into numpy arrays or torch tensors.
    pass


def get_data_loaders(path_to_csv, train_val_test=[0.8, 0.2, 0.2], batch_size=32):
  # First we create the dataset given the path to the .csv file
  dataset = SimpleDataset(path_to_csv)

  # Then, we create a list of indices for all samples in the dataset.
  dataset_size = len(dataset)
  indices = list(range(dataset_size))

  ## TODO: Rewrite this section so that the indices for each dataset split
  ## are formed.

  ## BEGIN: YOUR CODE
  train_indices = []
  val_indices = []
  test_indices = []
  ## END: YOUR CODE

  # Now, we define samplers for each of the train, val and test data
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

  test_sampler = SubsetRandomSampler(test_indices)
  test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

  return train_loader, val_loader, test_loader