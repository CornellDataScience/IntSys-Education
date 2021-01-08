import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_grad_descent_1d(h, grad_h, loss, dloss, x, y, grad_des,
                         x_support, y_support):
    """plot_grad_descent: plotting the gradient descent iterations.

    Generates the

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss: loss function that we will be optimizing on
    :type loss: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param dloss: the gradient of the loss function we are optimizing
    :type dloss: typing.Callable[
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
    :param grad_des: gradient descent blackbox optimizer
    :param x_support: range of x values to plot
    :type x_support: np.ndarray
    :param y_support: range of y values to plot
    :type y_support: np.ndarray
    :return: None
    :rtype: None
    """
    steps = 500
    _, thetas = grad_des(h, grad_h, loss, dloss, x, y, steps)

    fig, ax = plt.subplots()
    ax.set_xlim([x_support[0], x_support[1]])
    ax.set_ylim([y_support[0], y_support[1]])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt.title("Gradient Descent in 1D")
    line, = ax.plot([], [])
    scat = ax.scatter([], [], c="red")
    text = ax.text(x_support[0], y_support[0], "")

    potential_theta = np.linspace(x_support[0], x_support[1], 1000)  # .reshape((-1,1))
    potential_loss = [loss(h, grad_h, potential_theta[j].reshape((-1, 1)), x, y) for j in range(1000)]

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        global theta
        global loss_val
        # Gradient descent
        theta = thetas[(thetas.shape[0] // steps) * i]
        loss_val = loss(h, grad_h, theta, x, y)

        # Update the plot
        scat.set_offsets([[theta, loss_val]])
        text.set_text("Loss Value : {:.2f} Theta Value : {:.2f}".format(loss_val, theta[0, 0]))
        line.set_data(potential_theta, potential_loss)
        return line, scat, text

    ani = animation.FuncAnimation(fig, animate, steps,
                                  init_func=init, interval=500, blit=True)

    ani.save("gradDes_anim.gif", writer='imagemagick', fps=30)

    return None


def plot_linear_1d(h, grad_h, loss, dloss, x, y, grad_des, x_support, y_support):
    """plot_grad_descent: plotting the gradient descent iterations.

    Generates the

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss: loss function that we will be optimizing on
    :type loss: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param dloss: the gradient of the loss function we are optimizing
    :type dloss: typing.Callable[
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
    :param grad_des: gradient descent blackbox optimizer
    :param x_support: range of x values to plot
    :type x_support: np.ndarray
    :param y_support: range of y values to plot
    :type y_support: np.ndarray
    :return: None
    :rtype: None
    """
    import sys
    steps = 500
    _, thetas = grad_des(h, grad_h, loss, dloss, x, y, steps)

    fig, ax = plt.subplots()
    ax.set_xlim([min(x) - 0.5, max(x) + 0.5])
    ax.set_ylim([min(y) - 0.5, max(y) + 0.5])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt.title("Gradient Descent in 1D")
    line, = ax.plot([], [])
    scat = ax.scatter(x.reshape((-1,)), y.reshape((-1,)), c="red")
    text = ax.text(x_support[0], y_support[0], "")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        global theta
        global preds
        # Gradient descent
        theta = thetas[(thetas.shape[0] // steps) * i]
        x_range = np.arange(x_support[0], x_support[1], 0.01)
        preds = h(theta, x_range.reshape((-1, 1)))

        # Update the plot
        text.set_text("Theta Value : {:.2f}".format(theta[0, 0]))
        line.set_data(x_range, preds.reshape((-1,)))
        return line, scat, text

    ani = animation.FuncAnimation(fig, animate, steps,
                                  init_func=init, interval=500, blit=True)

    ani.save("linear_anim.gif", writer='imagemagick', fps=30)

    return None
