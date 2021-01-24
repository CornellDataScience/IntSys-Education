import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders
from typing import List, Union, Tuple


class SimpleNeuralNetModel(nn.Module):
    """SimpleNeuralNetModel [summary]
    
    [extended_summary]
    
    :param layer_sizes: Sizes of the input, hidden, and output layers of the NN
    :type layer_sizes: List[int]
    """
    def __init__(self, layer_sizes: List[int]):
        super(SimpleNeuralNetModel, self).__init__()
        # TODO: Set up Neural Network according the to layer sizes
        # The first number represents the input size and the output would be
        # the last number, with the numbers in between representing the
        # hidden layer sizes
        raise NotImplementedError()
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        raise NotImplementedError()


class SimpleConvNetModel(nn.Module):
    """SimpleConvNetModel [summary]
    
    [extended_summary]
    
    :param img_shape: size of input image as (W, H)
    :type img_shape: Tuple[int, int]
    :param output_shape: output shape of the neural net
    :type output_shape: tuple
    """
    def __init__(self, img_shape: Tuple[int, int], output_shape: tuple):
        super(SimpleConvNetModel, self).__init__()
        # TODO: Set up Conv Net of your choosing. You can / should hardcode
        # the sizes and layers of this Neural Net. The img_size tells you what
        # the input size should be and you have to determine the best way to
        # represent the output_shape (tuple of 2 ints, tuple of 1 int, just an
        # int , etc).
        raise NotImplementedError()
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        raise NotImplementedError()


if __name__ == "__main__":
    ## You can use code similar to that used in the LinearRegression file to
    # load and train the model.
    pass
