import matplotlib
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compute_error_for_line_given_points(b, m, points):
    # Initialize error at 0
    total_error = 0
    # For every point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # Starting points
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m
        b_gradient += -(2 / n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + b_current))
    # Update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    # Gradient descent
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def abline(slope, intercept, x_vals):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    # x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'red')


def run():
    # Collect our data, randomize it, and separate it into training and testing ndarrays:
    points = genfromtxt('cementVsCCS_FullDataSet.csv', delimiter=',')
    np.random.shuffle(points)
    training_points, testing_points = points[:900,:], points[900:,:]

    # separate the points into x and y tuples
    x_training_values, y_training_values = zip(*training_points)
    x_testing_values, y_testing_values = zip(*testing_points)

    # convert back to ndarrays
    x_ndarray_training = np.asarray(x_training_values)
    y_ndarray_training = np.asarray(y_training_values)
    x_ndarray_testing = np.asarray(x_testing_values)
    y_ndarray_testing = np.asarray(y_testing_values)

    # Defining hyper parameters:
    learning_rate = 0.00001
    initial_b = 0
    initial_m = 0
    num_iterations = 4000

    # Train our model:
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('After {0} iterations b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m,
                                                                      compute_error_for_line_given_points(b, m,
                                                                                                          points)))
    # plot the scatter plot of the training data and the linear fit, then show the plot
    plt.scatter(x_ndarray_training, y_ndarray_training, c="green")
    abline(b, m, x_ndarray_training)
    plt.show()

    # clear the plot of the training data
    plt.clf()

    # plot the scatter plot of the testing data and the linear fit, then show the plot
    plt.scatter(x_ndarray_testing, y_ndarray_testing, c="blue")
    abline(b, m, x_ndarray_training)
    plt.show()






if __name__ == '__main__':
    run()