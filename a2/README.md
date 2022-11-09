# Welcome to A2!

You made it through another onboarding lecture! Now, onto your second assignment: Implementing Linear and Logistic Regression!

## A Quick Refresher

We talked about two different modeling approaches in onboarding: Linear and Logistic Regression. Regression is the task of trying to model the patterns in a dataset - that is, we wanna be able to put in some inputs x and get some numerical output f(x). Linear Regression models a linear relationship between the input features x and the output f(x): where the output f(x) is a result of the sum of each feature of x (x_i) scaled by some scalar c_i. On the other hand, Logistic Regression, despite its name, is a classification algorithm that takes in features x and returns either 0 or 1. We'll be implementing these algorithms in A2.

(TL;DR: Linear Regression takes input numbers x and returns an output number f(x). Logistic Regression takes input numbers x and returns an output classification either 0 or 1.)

## PyWhat?

You'll be working on two files: `linear_regression.py` and `logistic_regression.py`. We'll walk through each file's TODOs and give you some tips and tricks for this assignment!

One of the big things we want to learn today is how to do things with PyTorch, a machine learning library for Python. For some ideas on how to use PyTorch, please refer to this article: https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

## `linear_regression.py`: The Real Regression 

We'll be building the model end to end, through TODOs 1 - 5. 

### `__init__`: Getting Started with TODO 1
 
PyTorch works by giving you the skeleton of a function that you'll be fleshing out. This is what the first line of this function does - it sets up the skeleton of your function. What you'll be doing here is tacking on some extra meat and bones on the skeleton that nn.Module gives you. You'll want to give the LinearRegressionModel object an nn.Linear layer here, built to fit the needs of the input task. Think of this step like getting a custom wrench crafted at the hardware store: you know the kind of tool you need (an nn.Linear layer) but you don't know what size nut you want the wrench to work for (the inputs and outputs.)

### `forward`: Creating A Prediction with TODO 2

One of the cool things with PyTorch and Python is that you can take a function set up in `__init__` and directly apply it to some inputs (okay, this is also the case with many other object-oriented languages, but I digress). If you can refer to the LinearRegressionModel object's linear layer, you can just treat it like a regular function and get the output for your model. 

### `data_transform`: If You Want To

It's a tool for you if you feel like you need it - for example, if your data has one feature that is messing things up, you can use this guy to remove it. For now, you can ignore this function. 

### `mse_loss`: Mean Squared Error Loss with TODO 3

You'll be implementing the MSE Loss formula in this function. As the name implies, MSE is the following formula: 

![MSE Loss Formula](https://miro.medium.com/max/640/1*-e1QGatrODWpJkEwqP4Jyg.png)

The key thing to notice here is that you'll be taking in and producing a torch.Tensor object. You can't do the same kind of matrix math you did in A1: instead, you'll have to use PyTorch's native Tensor operations to do the math for you. We recommend that you look to [this link](https://pytorch.org/docs/stable/tensors.html#initializing-and-basic-operations) to learn how to create a new Tensor and [these PyTorch functions](https://pytorch.org/docs/stable/torch.html#math-operations) to help you calculate the MSE.

### `mae_loss`: Mean Absolute Error Loss with TODO 4

You'll be implementing the MAE Loss formula in this function. As the name implies, MAE is the following formula:

![MAE Loss Formula](https://miro.medium.com/proxy/1*OVlFLnMwHDx08PHzqlBDag.gif)

Like before, you'll be using PyTorch functions to create a (1, 1) Tensor containing the MAE Loss value. Again, we recommend that you look to [this link](https://pytorch.org/docs/stable/tensors.html#initializing-and-basic-operations) to learn how to create a new Tensor and [these PyTorch functions](https://pytorch.org/docs/stable/torch.html#math-operations) to help you calculate the MAE. 

### `__main__`: Bringing It All Together with TODO 5

You'll be wrapping everything together in this function. We strongly recommend referring to [this article](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817) to learn about how to run a PyTorch model. We've also given you `plotting.py` like last time, so you can see how your model's done. 

## `logistic_regression.py`: Classification With PyTorch

You'll be implementing `LogisticRegressionModel`. Then, you'll be implementing `logistic_loss` and setting up a `__main__` function to run the models and generate predictions. 

### `LogisticRegressionModel`: Just Like `LinearRegressionModel`

Similarly to `LinearRegressionModel`, you'll be setting up an `__init__` function and a `forward` function. Much like before, you'll be building onto the nn.Module skeleton provided by PyTorch. This time, you'll probably need more than one calculation - one to calculate the linear output f(x) (associated with the weights you'll be learning) and one to calculate the Sigmoid function that will predict the classification of the object. We recommend using the same nn.Linear layer from `LinearRegressionModel` to get the first item. Then, we recommend applying an nn.Sigmoid function to the output to get P(Y=1) or P(Y=0). We recommend checking out [this article](https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be) for help.

### `MultinomialRegressionModel`: Bonus

If you wanna give this one a go, feel free to try. Not part of the assignment though. 

### `logistic_loss`: Loss for Logistic Regression

You'll be using PyTorch tools to create the Logistic Loss function: 

![Log Loss Formula](https://miro.medium.com/max/4800/1*CQpbokNStSnBDA9MdJWU_A.png)

Like before, you'll be using PyTorch functions to create a (1, 1) Tensor containing the Logistic Loss value. Again, we recommend that you look to [this link](https://pytorch.org/docs/stable/tensors.html#initializing-and-basic-operations) to learn how to create a new Tensor and [these PyTorch functions](https://pytorch.org/docs/stable/torch.html#math-operations) to help you calculate the Logistic Loss. 

### `cross_entropy_loss`: Another Bonus

Feel free to look into this one if you'd like!

### `__main__`: Putting it all together

You'll be wrapping everything together in this function. We strongly recommend referring to [this article](https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be) to learn about how to run the Logistic Regression PyTorch model. We've also given you `plotting.py` like last time, so you can see how your model's done.

Good luck! Please let us know if you have any questions or concerns!
