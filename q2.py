import numpy as np
import matplotlib.pyplot as plt

"""
q2.py

This program builds a logistic regression model for binary classification of given dataset

@author: Anushree Sitaram Das (ad1707)
"""

def getData(filename):
    """
    load dataset from csv file
    :param filename:
    :return:
    """
    data = np.genfromtxt(filename, delimiter=",", skip_header=1,dtype=( float, float, "|S10"), names=["MFCCs_10",	"MFCCs_17",	"Species"])
    return data


def sigmoid(x):
    """
    Returns value between 0 and 1
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def costFunction(X, y, theta):
    """
    Calculate error of model output compared with actual output
    :param X:           features dataset
    :param y:           output vector
    :param theta:       initial weights
    :return:
    """
    m = len(y)
    fTheta = sigmoid(X @ theta)
    cost = (1/m)*(((-y).T @ np.log(fTheta))-((1-y).T @ np.log(1-fTheta)))
    return cost


def gradient_descent(X, y, theta, iterations):
    """
    Find optimal weights
    :param X:           features dataset
    :param y:           output vector
    :param theta:       initial weights
    :param iterations:  number of iterations
    :return:            optimal weights and previous weights history for plotting
    """
    m = len(y)
    # stores history of weights
    theta_all = np.zeros(shape=(theta.shape[0],1))
    # stores history of cost
    cost_history = np.zeros((iterations,1))

    # find optimal weights
    for i in range(iterations):
        fTheta = sigmoid(X.dot(theta))
        cost_history[i] = costFunction(X,y,theta)
        theta = theta - (1/m) * (X.T.dot(fTheta - y))
        theta_all = np.concatenate((theta_all, theta), 1)

    return (cost_history,theta, theta_all)


def scatterPlot(data,a,theta,theta_all=None):
    """
    Plot data and decision boundary
    :param data:        dataset
    :param a:           dataset name
    :param theta:       parameters for decision boundary
    :param theta_all:   previous parameters for decision boundary
    :return:
    """
    # load features
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # assign color for each input according to it specie
    colors = []
    for specie in data["Species"]:
        s = specie.decode("utf-8")
        if s == 'HylaMinuta':
            colors.append('red')
        else:
            colors.append('blue')

    # generate scatter plot
    plt.scatter(feature1, feature2, s=5, alpha=0.7, color=colors)
    plt.xlabel('MFCCs_10')
    plt.ylabel('MFCCs_17')
    plt.title('Scatter Plot for '+a)

    # plot previous decision boundaries
    if theta_all is not None:
        for i in range(10,len(theta_all[0]),100):
            slope = -(theta_all[0][i] / theta_all[2][i]) / (theta_all[0][i] / theta_all[1][i])
            intercept = -theta_all[0][i] / theta_all[2][i]
            x_vals = np.linspace(np.amin(feature1), np.amax(feature1))
            y_vals = (slope * x_vals) + intercept
            plt.plot(x_vals, y_vals, 'm-', alpha = 0.3)

    # final decision boundary
    slope = -(theta[0] / theta[2]) / (theta[0] / theta[1])
    intercept = -theta[0] / theta[2]
    x_vals = np.linspace(np.amin(feature1),np.amax(feature1))
    y_vals =  (slope*x_vals) + intercept
    plt.plot(x_vals, y_vals, '--')

    # Save the figure and show
    # plt.savefig('logistic_regression_' + a + '.png')
    plt.show()


def logisticRegression(data,a):
    """
    Apply Logistic Regression on given dataset to get optimal parameters
    :param data:    dataset
    :param a:       dataset name
    :return: optimal weights(parameters)
    """
    category = data["Species"]
    m = len(category)
    y = np.zeros((m, 1))

    # convert classes to 0 and 1
    # 1 if class is HylaMinuta else 0
    for i in range(len(category)):
        s = category[i].decode("utf-8")
        if s == 'HylaMinuta':
            y[i][0] = 1

    # dataset of features
    # column of ones is added for bias
    X = np.column_stack((np.ones((m, 1)), data["MFCCs_10"], data["MFCCs_17"]))
    n = np.size(X, 1)
    # weights for network
    theta = np.zeros((n, 1))

    # interations to train model
    iterations = 1500

    # train model to get optimal weights
    (cost_history, theta_optimal, theta_all) = gradient_descent(X, y, theta, iterations)

    # plot cost_history
    plt.plot(list(range(iterations)), cost_history, '-r')
    plt.title('Cost Function History for '+a)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    # plt.savefig('cost_history_'+a+'.png')
    plt.show()

    print("Optimal Parameters are: \n", theta_optimal, "\n")

    # plot how decision boundary evolved over iterations
    scatterPlot(data, a, theta_optimal, theta_all)

    return theta_optimal


def predict(X, theta):
    result = np.round(sigmoid(X @ theta))
    if result > 0.5:
        return 'HylaMinuta'
    else:
        return 'HypsiboasCinerascens'


if __name__ == "__main__":

    # load data
    sampledata = getData("Frogs-subsample.csv")
    data = getData("Frogs.csv")

    # create binary classifier for the data in each file using a single logistic regressor
    # and get the weights
    # theta_optimal1 = logisticRegression(sampledata,'Frogs-subsample_training')
    # theta_optimal2 = logisticRegression(data,'Frogs_training')

    # this file contains the saved parameters for the datasets
    f = open("OptimalParameters.txt", "r")
    parameters = []
    for x in f:
        parameters.append(float(x))

    # scatter plot with decision boundaries
    # scatterPlot(sampledata, 'Frogs-subsample', theta_optimal1)
    # scatterPlot(data, 'Frogs', theta_optimal2)

    feature1 = float(input('Enter MFCCs_10:'))
    feature2 = float(input('Enter MFCCs_17:'))

    X = np.array([1,feature1, feature2]).T
    print('Class:',predict(X,parameters))




