import csv
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Here in this file, you should define functions to try out different encodings.
# Some options:
#   1) Bag of words. You don't need any library for this, it's simple enough to
#       implement on your own.
#   2) Word embeddings. You can use spacy or Word2Vec (or others, but these are good
#       starting points). Spacy will give you better embeddings, since they are 
#       already defined. Word2Vec will create embeddings based on your dataset.

## Document the choices you make, including if you pre-process words/tokens by 
# stemming, using POS (parts of speech info) or anything else. 

## Create your own files to define Logistic regression/ Neural networks to try
# out the performace of different ML algorithms. It'll be up to you to evaluate
# the performance.


class SentimentDataset(Dataset):
    """SentimentDataset [summary]
    
    [extended_summary]
    
    :param path_to_data: Path to dataset directory
    :type path_to_data: str
    """
    def __init__(self, path_to_data):
        ## TODO: Initialise the dataset given the path to the dataset directory.
        ## You may want to include other parameters, totally your choice.
        pass

    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        pass

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        ## TODO: This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input 
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## sepcified, and pply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.

        pass


def get_data_loaders(path_to_pkl, 
                     path_to_labels,
                     train_val_test=[0.8, 0.2, 0.2], 
                     batch_size=32):
  """
  You know the drill by now.
  """
  pass