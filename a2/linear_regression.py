import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders

class LinearRegressionModel(nn.Module):
    """LinearRegressionModel is the linear regression regressor.

    This class handles only the standard linear regression task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    :param loss_fn: The loss function that is used to calculate "cost"
    :type loss_fn: typing.Callable[[torch.Tensor, torch.Tensor],torch.Tensor]
    """
    def __init__(self, num_param, loss_fn):
        ## TODO 1: Set up network
        pass

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = LinearRegressionModel(1, mse_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 2: Implement the linear regression on sample x
        pass


def mse_loss(output, target):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left( x_n - y_n \\right)^2,

    where :math:`N` is the batch size. 

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor, np.ndarray]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 3: Implement Mean-Squared Error loss. 
    # Use PyTorch operations to return a PyTorch tensor
    pass

def mae_loss(output, target):
    """Creates a criterion that measures the mean absolute error (l1 loss)
    between each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left| x_n - y_n \\right|,

    where :math:`N` is the batch size. 

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor, np.ndarray]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 4: Implement L1 loss. Use PyTorch operations.
    # Use PyTorch operations to return a PyTorch tensor.
    pass


if __name__ == "__main__":
    ## Here you will want to create the relevant dataloaders for the csv files for which 
    ## you think you should use Linear Regression. The syntax for doing this is something like:
    # Eg:
    # train_loader, val_loader, test_loader =\
    #   get_data_loaders(path_to_csv, train_val_test=[YOUR TRAIN/VAL/TEST SPLIT], batch_size=YOUR BATCH SIZE)


    ## Now you will want to initialise your Linear Regression model, using something like
    # Eg:
    # model = LinearRegressionModel(...)


    ## Then, you will want to define your optimizer (the thing that updates your model weights)
    # Eg:
    # optimizer = optim.[one of PyTorch's optimizers](model.parameters(), lr=0.01)


    ## Now, you can start your training loop:
    # Eg:
    # model.train()
    # for t in range(TOTAL_TIME_STEPS):
    #   for batch_index, (input_t, y) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #
    #     preds = Feed the input to the model
    #
    #     loss = loss_fn(preds, y)  # You might have to change the shape of things here.
    #     
    #     loss.backward() 
    #     optimizer.step()
    #     
    ## Don't worry about loss.backward() for now. Think of it as calculating gradients.

    ## And voila, your model is trained. Now, use something similar to run your model on
    ## the validation and test data loaders:
    # Eg: 
    # model.eval()
    # for batch_index, (input_t, y) in enumerate(val/test_loader):
    #
    #   preds = Feed the input to the model
    #
    #   loss = loss_fn(preds, y)
    #
    ## You don't need to do loss.backward() or optimizer.step() here since you are no
    ## longer training.

    pass
