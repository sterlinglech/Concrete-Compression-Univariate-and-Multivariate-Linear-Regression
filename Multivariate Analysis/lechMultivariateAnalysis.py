import numpy as np
import pandas as pd
import matplotlib
from sklearn import metrics
from sklearn.metrics import r2_score
matplotlib.use('TkAgg')
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

def calculate_r2(y_testing_param, X_testing_param, theta_vector_param, column_string_index):

    TestingDataFrameColumn = X_testing_param.loc[:, [column_string_index]]
    TestingArray = TestingDataFrameColumn.to_numpy()
    predicted_y_array = TestingArray * theta_vector_param

    return metrics.r2_score(y_testing_param, predicted_y_array)
def cost_function(X_param, y_param, theta_param):
    m = y_param.size
    error = np.dot(X_param, theta_param.T) - y_param
    cost = 1 / (2 * m) * np.dot(error.T, error)
    mse = np.mean((np.dot(X_param, theta_param) - y_param) ** 2) / 2
    return cost ,error, mse

def gradient_descent(X_param, y_param, theta_param, alpha, iters):
    cost_array = np.zeros(iters)
    m = y_param.size
    for i in range(iters):
        cost, error, mse = cost_function(X_param, y_param, theta_param)
        theta_param = theta_param - (alpha * (1 / m) * np.dot(X_param.T, error))
        cost_array[i] = cost
    return theta_param, cost_array

def run():
    # Import data
    data = pd.read_csv('Concrete_Data.csv')

    # Extract data into X and y
    fullDataMatrix = data[['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)',
              'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)', 'Concrete compressive strength']]

    # randomize the data
    fullDataMatrix.sample(frac=1)

    # separate into x and y dataframes
    X = fullDataMatrix[['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)',
              'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)']]

    y = fullDataMatrix['Concrete compressive strength']

    # separate our features and response samples into training and testing data
    X_training = X.head(900)
    X_testing = X.tail(130)

    y_training = y.head(900)
    y_testing = y.tail(130)

    # Normalize our features
    X_training = (X_training - X_training.mean()) / X_training.std()
    X_testing = (X_testing - X_testing.mean()) / X_testing.std()

    # Add a 1 column to the training features to the start to allow vectorized gradient descent
    X_training = np.c_[np.ones(X_training.shape[0]), X_training]

    # Set hyperparameters to train the model with
    alpha = 0.001
    iterations = 8000

    # Initialize Theta Values to 0
    theta_vector = np.zeros(X_training.shape[1])
    initial_cost, _, initial_mse = cost_function(X_training, y_training, theta_vector)

    # initial values of MSE and Thetas/Coefficients
    print('With initial theta values of {0}, cost error is {1}, mse is {2}'.format(theta_vector, initial_cost, initial_mse))

    # Run Gradient Descent
    theta_vector, cost_num = gradient_descent(X_training, y_training, theta_vector, alpha, iterations)


    # calculate the R2 scores for all the testing features
    print('The R2 score for the cement component testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[0], 'Cement (component 1)(kg in a m^3 mixture)')))
    print('The R2 score for the Blast Furnace Slag component testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[1], 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)')))
    print('The R2 score for the Fly Ash component testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[2], 'Fly Ash (component 3)(kg in a m^3 mixture)')))
    print('The R2 score for the Water component testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[3], 'Water  (component 4)(kg in a m^3 mixture)')))
    print('The R2 score for the Superplasticizer (component 5)(kg in a m^3 mixture) testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[4], 'Superplasticizer (component 5)(kg in a m^3 mixture)')))
    print('The R2 score for the Coarse Aggregate  (component 6)(kg in a m^3 mixture) testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[5], 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)')))
    print('The R2 score for the Fine Aggregate (component 7)(kg in a m^3 mixture) testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[6], 'Fine Aggregate (component 7)(kg in a m^3 mixture)')))
    print('The R2 score for the Age (day) testing data is: {0}'.format(calculate_r2(y_testing, X_testing, theta_vector[7], 'Age (day)')))




if __name__ == "__main__":
    run()