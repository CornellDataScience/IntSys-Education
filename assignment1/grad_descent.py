"""Gradient Descent Assignment for CDS Intelligent Systems."""

import typing

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Example Hypothesis Functions
# ============================================================================


def linear_h(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """linear_h: The linear hypothesis regressor.

    :param theta: parameters for our linear regressor
    :type theta: np.ndarray
    :param x: input that model is predicting; shape is (samples, features)
    :type x: np.ndarray
    :return: The predictions of our model on inputs X
    :rtype: np.ndarray
    """
    return theta @ x


def linear_grad_h(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """linear_h: The gradient of the linear hypothesis regressor.

    :param theta: parameters for our linear regressor
    :type theta: np.ndarray
    :param x: input that model is predicting; shape is (samples, features)
    :type x: np.ndarray
    :return: The gradient of our linear regressor
    :rtype: np.ndarray
    """
    return x


def parabolic_h(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """parabolic_h: The parabolic hypothesis regressor.

    :param theta: parameters for our parabolic regressor
    :type theta: np.ndarray
    :param x: input that model is predicting; shape is (samples, features)
    :type x: np.ndarray
    :return: The predictions of our model on inputs X
    :rtype: np.ndarray
    """
    return theta @ (x ** 2)


def parabolic_grad_h(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """parabolic_grad_h: The gradient of the parabolic hypothesis regressor.

    :param theta: parameters for our parabolic regressor
    :type theta: np.ndarray
    :param x: input that model is predicting; shape is (samples, features)
    :type x: np.ndarray
    :return: The gradient of our parabolic regressor
    :rtype: np.ndarray
    """
    return x ** 2


# Add your own hypotheses if you want


def plot_grad_descent():
    """plot_grad_descent: plotting the gradient descent iterations.

    Write an extended summary

    :return: None
    :rtype: None
    """
    return None


def loss_f1(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """loss_f1 returns the loss for special function f1.

    This function is for demonstration purposes, since it ignores
    data points x and y.

    :param h: hypothesis function that is being used
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters for our model, must be of shape (2,)
    :type theta: np.ndarray of shape (-1, 2)
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: Return the function evaluation of theta, x, y
    :rtype: int or np.ndarray of shape (theta.shape[1],)
    """
    theta = np.reshape(theta, (-1, 2))
    w1 = theta[:, 0]
    w2 = theta[:, 0]
    return (
        -2 * np.exp(-((w1 - 1) * (w1 - 1) + w2 * w2) / 0.2)
        + -3 * np.exp(-((w1 + 1) * (w1 + 1) + y * y) / 0.2)
        + w1 * w1
        + w2 * w2
    )


def grad_loss_f1(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """grad_loss_f1 returns the gradients for the loss of the f1 function.

    This function is for demonstration purposes, since it ignores
    data points x and y.

    :param h: The hypothesis function that predicts our output given weights
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: The gradient function of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters for our model.
    :type theta: np.ndarray of shape (-1, 2)
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: gradients for the loss function along the two axes
    :rtype: np.ndarray
    """
    theta = np.reshape(theta, (-1, 2))
    w1 = theta[:, 0]
    w2 = theta[:, 0]
    step = 1e-7
    grad_w1 = (loss_f1(w1 + step, w2) - loss_f1(w1, w2)) / step
    grad_w2 = (loss_f1(w1, w2 + step) - loss_f1(w1, y)) / step
    return np.array((grad_w1, grad_w2))


def l2_loss(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """l2_loss: standard l2 loss.

    The l2 loss is defined as (h(x) - y)^2. This is usually used for linear
    regression in the sum of squares.

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters of our hypothesis fucntion
    :type theta: np.ndarray
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: The l2 loss value
    :rtype: float
    """
    return np.sum(np.square((h(theta, x) - y)))


def grad_l2_loss(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """grad_l2_loss: The gradient of the standard l2 loss.

    The gradient of l2 loss is given by d/dx[(h(x) - y)^2] which is
    evaluated to 2*(h(x) - y)*h'(x).

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters of our hypothesis fucntion
    :type theta: np.ndarray
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: The l2 loss value
    :rtype: float
    """
    return np.sum(2 * (h(theta, x) - y) * grad_h(theta, x))


# ============================================================================
# YOUR CODE GOES HERE:
# ============================================================================


def grad_descent(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    loss_f: typing.Callable[
        [
            typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
            typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        np.ndarray,
    ],
    grad_loss_f: typing.Callable[
        [
            typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
            typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        np.ndarray,
    ],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """grad_descent: gradient descent algorithm on a hypothesis class.

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match
    :param y: np.ndarray
    :return: The ideal parameters
    :rtype: np.ndarray
    """
    # TODO 1:
    return np.zeros((1,))


def stochastic_grad_descent():

    # TODO 2
    return np.zeros((1,))


def minibatch_grad_descent():

    # TODO 3
    return np.zeros((1,))


def matrix_gd():

    # TODO 4
    return np.zeros((1,))


def matrix_sgd():

    # TODO 5
    return np.zeros((1,))


def matrix_minibatch_gd():

    # TODO 6
    return np.zeros((1,))


if __name__ == "__main__":

    pass
