import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from a2.data_loader import get_data_loaders

class LinearRegressionModel(nn.Module):
  def __init__(self, num_param, loss_fn):
    ## TODO 1: Set up network
    pass

  def forward(self, x):
    ## TODO 2: Implement the linear regression on sample x
    pass


def mse_loss(output, target):
  ## TODO 3: Implement Mean-Squared Error loss. 
  # Use PyTorch operations to return a PyTorch tensor
  pass

def abs_loss(output, target):
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
  