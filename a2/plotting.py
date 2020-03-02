import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import typing


def plot_binary_logistic_boundary(logreg, X, y, xlim, ylim):
    """Plots the boundary given by the trained logistic regressor
    
    :param logreg: Logistic Regrssor model
    :type logreg: logistic_regression.LogisticRegressionModel
    :param X: The features and samples used to train
    :type X: np.ndarray
    :param y: The labels for the classification task
    :type y: np.ndarray
    :param xlim: min and max :math:`x_1` values for the plot
    :type xlim: typing.Tuple[int, int]
    :param ylim: min and max :math:`x_2` values for the plot
    :type ylim: typing.Tuple[int, int]
    """

    xx, yy = np.mgrid[xlim[0]:xlim[1]:.01, ylim[0]:ylim[1]:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = logreg(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                        vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:,0], X[:, 1], c=y, s=50,
            cmap="RdBu", vmin=-.2, vmax=1.2,
            edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
        xlim=xlim, ylim=ylim,
        xlabel="$X_1$", ylabel="$X_2$")

    plt.show()

    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    ax.scatter(X[:,0], X[:, 1], c=y, s=50,
            cmap="RdBu", vmin=-.2, vmax=1.2,
            edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
        xlim=xlim, ylim=ylim,
        xlabel="$X_1$", ylabel="$X_2$")
    
    plt.show()


def plot_linear_1D(linreg, X, y, xlim, ylim):
    """Plots the best plane given by the trained linear regressor
    
    :param logreg: Logistic Regrssor model
    :type logreg: linear_regression.LinearRegressionModel
    :param X: The features and samples used to train
    :type X: np.ndarray
    :param y: The labels for the regression task
    :type y: np.ndarray
    :param xlim: min and max :math:`x` values for the plot
    :type xlim: typing.Tuple[int, int]
    :param ylim: min and max :math:`y` values for the plot
    :type ylim: typing.Tuple[int, int]
    """
    pass
    