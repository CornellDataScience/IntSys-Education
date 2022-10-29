# Welcome to A1!

Congratulations on making it through your very first onboarding lecture! Now, it's time to put what you've learned into practice and implement your very own Gradient Descent algorithm!

## A Quick Refresher

Gradient Descent works by giving your model feedback on a set of predictions using the gradient of its loss function. Here's how it goes: 
* Your model makes a prediction(s) on some input(s).
* The loss function tells you how good or bad your prediction(s) was.
* The gradient of the loss function tells your model where to go to make a better prediction on the specific input(s) you gave the model.
* You improve your model based on the gradient of the loss function.
* You give your model some more input(s) and repeat the process until you're satisfied.

## What's in A1?

Your work will be done in the `grad_descent.py` file. Fitting, I know. Your main task will be to set up a training loop - a way to keep feeding the model new inputs and improving it based on those inputs. We've given you a few handy-dandy functions to get your feet on the ground - your job will be to put it all together!

## The Freebies

Here, we'll tell you about the various freebies we've given you - what they do, how they work, and any other little tidbits we think would help you in your task. Generally, they fall into two categories: hypothesis functions and loss functions.

### Hypothesis Functions?

Hypothesis functions are the base function you'll be improving with this model. Think of your Hypothesis function as the rough shape you want your model to follow. In the future, we won't need these guys - we'll talk about much more flexible models that can take on just about any shape you can come up with - but for now, we'll be using these guys as a starting point for our gradient descent algorithm. When you make your gradient descent function, you don't need to worry about specifying which Hypothesis function to use - for your purposes, every Hypothesis function is the same: it takes in a theta and an input and spits out some predictions. 

### Theta?

Remember in Algebra, when they taught you a bunch of different ways to write a line equation (most of which are dumb and useless, and one of which is y = mx + b)? We learned that increasing m made the line be steep, making m negative made it go down instead of up, so on and so forth. In this case, we can think of the m as the line's Theta - a parameter that adjusts the line so it's going in the direction you want it to go. Theta is what you're going to try and learn: you're  going to try and figure out what m makes the line fit your data best. 

Importantly, Theta can't fundamentally change anything about your hypothesis function. In the Algebra example, no value of m out there is gonna make your y = mx + b turn into a circle, or a triangle, or whatever this monstrosity is:

![We were so preoccupied by whether we should, that we didn't stop to think whether we should.](https://qph.cf2.quoracdn.net/main-qimg-a021ded8906493c19279b56fef5e91fa)

The theta you learn can basically do two things: stretch your function out and/or flip it around. You'll be showing your Theta where to go using the gradient of the loss function.

### `linear_h` - Baby's First H-Function

It's exactly what you think it is - a very basic y = mx + b type function. What's special about this function is that it can do y = mx + b in multiple dimensions. In 2D, it's your classic straight line. In 3D, it's a flat plane. In anything-more-than-3D, it's what we call a hyperplane - a hard-to-visualize thing, but it's easiest to think of it in normal 2D or 3D terms: a flat thing that can split a space into two halves follwing pretty boring rules. 

(For this assignment, you probably don't need to care about the 4+D - just focus on the 2D and 3D versions, and you should be good!)

### `linear_grad_h` - Baby's First Huhh?

I know what you're thinking - why's it just an x? I thought we were doing gradient or something! You're not going crazy - this really is the gradient of the Linear Hypothesis Function! Remember that we're trying to make the Theta better, so we want to use the gradient with respect to Theta - else, we'd be racking our heads over how to make the inputs better, which is 1) silly and 2) bad data science! Regardless, you probably won't be directly using this function - it's going to be fed into one of our loss functions down the line.

### `parabolic_h` and `parabolic_grad_h` - More of the Same

These guys are parabolic versions of the `linear_h` and `linear_grad_h`. After all, Theta can't make a line into a circle - or, in this case, a parabola - so we made one for you! That way, you can make a model that fits parabola-shaped data and not feel silly when you're trying to make a straight line work out (it probably won't, unfortunately). They work very much the same as their linear counterparts - in your case, they're interchangeable - so long as you make sure you use the right grad_h function during testing!

### Loss Functions?

Like we said in lecture, you have to have a way to tell the model how it's doing. That's what the loss function does - it spits out a number telling you how good or bad the model's doing. Then, using the magic of calculus, we can use this one number to tweak every single item on our theta to better fit our data.

### `loss_f1` and `grad_loss_f1` - Some Loss Functions For Show

We've left a neat loss function for show - `loss_f1` and its counterpart `grad_loss_f1`. They show you how we might make a loss function reflect a model's F1 score while still being able to differentiate it (and thus, get cool information about how to make our model better!). It's not an easy problem to solve - the original F1 function isn't differentiable, so we had to make some adjustments - but we left it here because the person who made this assignment (shoutout to Samar, one of our old subteam leads!) is pretty smart, probably worked hard on this, and it's neat. 

Basically, for A1 purposes, you don't need to use these functions - we just liked Samar so we left these here. 

### `l2_loss` - Your Loss Function!

This is the loss function you'll be using to tell your model how well (or how poorly) it's doing! It's a pretty straightforward function - it takes the difference between what you predict y should be and what y actually is and squares that difference. It's a great way to tell you how your model is doing - everything it spits out will be positive and it's differentiable, so it's great for our purposes! It takes in the Hypothesis Function and its Gradient, the Theta, and a set of inputs and outputs (a lot, I'm sure!) and spits out how well the model (the Theta applied to the Hypothesis Functions) did in one number. In A1, we mostly use this function as a way to tell ourselves how well our model is doing - our Gradient Loss Function is independent of this guy.

### `grad_l2_loss` - Where The Magic Happens!

This function is where the magic happens! Its job is to give you the gradient of the loss function with respect to Theta - a lot of words to say that it tells you how to improve each part of Theta to get better results with the inputs you gave it! Importantly, you don't directly give this function the `l2_loss` - instead, you directly give it the inputs you gave the loss function (H Function, Gradient of H Function, Theta, Inputs, Outputs) and it spits out a Theta-shaped vector that tells your Theta where to go!

## Your Mission, Should You Choose To Accept It...

Your job will be to put all of these pieces together! We've described several different training approaches that build off of the basic concepts of Gradient Descent - you'll be building training loops for all of them and testing them out!

Since the heart of the training approaches is fundamentally the same (you'll be doing Gradient Descent every time!) we recommend that you reuse as much code as you can and only change the bits you need to change.

In each implementation, you'll be designing the same thing: a training loop that improves the model every time.

### Training Loop

The training loop will basically pull some training data and update the model until you tell it to stop. It's up to you to decide when you want your loop to stop, how big you want your model's steps to be, and how you want to decide what training data to give your model. How much training data you give depends on which implementation you're working on...

### TODO 1: OG Gradient Descent

This implementation is very careful - it looks at every single item in your training set before updating.

### TODO 2: Stochastic Gradient Descent

This implementation is the opposite of the first implementation - it only looks at one training sample every time it updates.

### TODO 3: Minibatch Gradient Descent

This is the Goldilocks implementation - it looks at some of the training data before updating. How much you want your model to look at is up to you!

### TODO 4, 5, and 6: Matrix Math iMplementation

Here, you re-implement the three GD approaches from TODO 1, 2, and 3 using Numpy to make the matrix math easier. 

## Testing

The most important part of any programming assignment! Once you think you've got an implementation working, pull up your terminal, go into the `a1` folder, and run this command:

* If you're on Mac: `python3 grad_descent.py`
* Otherwise: `python grad_descent.py`

It'll create a nice graph (and a nice GIF!) showing you how your GD implementation did! By default, it tests the GD implementation for TODO 1 - to test any other TODO, replace lines 467 and 478 with the name of the function you're trying to test.

Good luck and happy descending!
