import numpy as np
import matplotlib.pyplot as plt
import typing

## ============================================================================
## Example Hypothesis Functions
## ============================================================================


def linear_h(theta, x):
  return theta @ x


def linear_grad_h(theta, x):
  return x


def parabolic_h(theta, x):
  return theta @ (x ** 2)


def parabolic_grad_h(theta, x):
  return x**2


## Add your own hypotheses if you want


def loss_f1(h, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """loss_f1 returns the loss for special function f1.
  
  This function is for demonstration purposes, since it ignores
  data points x and y.
  
  :param theta: The parameters for our model, must be of shape (2,)
  :type theta: np.ndarray of shape (-1, 2)
  :param x: A matrix of samples and their respective features.
  :type x: np.ndarray of shape (samples, features)
  :param y: The expected targets our model is attempting to match
  :type y: np.ndarray of shape (samples,)
  :return: Return the function evaluation of theta, x, y
  :rtype: int or np.ndarray of shape (theta.shape[1],)
  """
  theta = np.reshape(theta, (-1,2))
  w1 = theta[:, 0]
  w2 = theta[:, 0]
  return -2 * np.exp(-((w1 - 1) * (w1 - 1) + w2 * w2) / .2) +\
         -3 * np.exp(-((w1 + 1) * (w1 + 1) + y * y) / .2) + w1 * w1 + w2 * w2


def grad_loss_f1(h: typing.Callable[np.ndarray, np.ndarray], 
                 grad_h: typing.Callable[np.ndarray, np.ndarray], 
                 theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """grad_loss_f1 returns the gradients for the loss of the f1 function.
  
  This function is for demonstration purposes, since it ignores
  data points x and y.
  
  :param h: The hypothesis function that predicts our output given weights
  :type h: typing.Callable[np.ndarray, np.ndarray]
  :param grad_h: The gradient function of our hypothesis function
  :type grad_h: typing.Callable[np.ndarray, np.ndarray]
  :param theta: The parameters for our model.
  :type theta: np.ndarray of shape (-1, 2)
  :param x: A matrix of samples and their respective features.
  :type x: np.ndarray of shape (samples, features)
  :param y: The expected targets our model is attempting to match
  :type y: np.ndarray of shape (samples,)
  :return: gradients for the loss function along the two axes
  :rtype: np.ndarray
  """
  theta = np.reshape(theta, (-1,2))
  w1 = theta[:, 0]
  w2 = theta[:, 0]
  step = 1e-7
  grad_w1 = (loss_f1(w1 + step, w2) - loss_f1(w1, w2)) / step
  grad_w2 = (loss_f1(w1, w2 + step) - loss_f1(w1, y)) / step
  return np.array((grad_w1, grad_w2))


def l2_loss(h, grad_h, theta, x, y):
  """l2_loss: standard l2 loss
  
  The l2 loss is defined as (h(x) - y)^2. This is usually used for linear
  regression in the sum of squares.
  
  :param h: [description]
  :type h: [type]
  :param theta: [description]
  :type theta: [type]
  :param x: [description]
  :type x: [type]
  :param y: [description]
  :type y: [type]
  :return: [description]
  :rtype: [type]
  """
  return np.sum(np.square((h(theta, x) - y)))


def grad_l2_loss(h, grad_h, theta, x, y):
  return np.sum(2 * (h(theta, x) - y) * grad_h(theta, x))


## ============================================================================
## YOUR CODE GOES HERE:
## ============================================================================

def grad_descent(theta, loss_f, grad_loss_f):
  
  ## TODO 1
  pass

def stochastic_grad_descent():

  ## TODO 2
  pass 

def minibatch_grad_descent():

  ## TODO 3
  pass

def matrix_gd():
  
  ## TODO 4
  pass

def matrix_sgd():
  
  ## TODO 5
  pass

def matrix_minibatch_gd():

  ## TODO 6
  pass

if __name__ == "__main__":
  
  pass