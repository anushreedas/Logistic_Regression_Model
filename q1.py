import numpy as np
import matplotlib.pyplot as plt

"""
q1.py

This program generates plots for visualisation of data and calculates statistics of data 

@author: Anushree Sitaram Das (ad1707)
"""

def getData(filename):
    """
    Load data from given file path using numpy and return it
    :param filename: path to csv file containing data
    :return: data from the csv file
    """
    return np.genfromtxt(filename, delimiter=",", skip_header=1,dtype=( float, float, "|S10"), names=["MFCCs_10",	"MFCCs_17",	"Species"])


def scatterPlot(data,a):
    """
    scatter plot of the ‘raw’ data (both features, separate color for each class)
    :param data:    raw data
    :param a:       filename
    :return:
    """
    # load features from the data
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

    # Save the figure and show
    # plt.savefig('scatter_plot_' + a + '.png')
    plt.show()

def histogramPlot(data,x):
    """
    generate histograms for given input features (1 per feature)
    :param data:    raw data
    :param x:       file name
    :return:
    """
    # load features from the data
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # generate histogram for each feature and display it side by side
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
    fig.suptitle('Histogram for ' + x)
    ax1.hist(feature1, bins=20)
    ax1.set_title('Feature 1 - MFCCs_10')
    ax2.hist(feature2, bins=20)
    ax2.set_title('Feature 2 - MFCCs_17')
    ax1.set(xlabel='MFCCs', ylabel='')
    ax2.set(xlabel='MFCCs', ylabel='')

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('histogram_' + x + '.png')
    plt.show()

def lineGraph(data,a):
    """
    sort feature values and
    generate line graphs (1 per feature)
    :param data:    raw data
    :param a:       filename
    :return:
    """
    # values on x-axis
    x = range(len(data["MFCCs_10"]))

    # load features from the data
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # sort features
    feature1.sort()
    feature2.sort()

    # generate line graphs for each feature and display side by side
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
    fig.suptitle('Line Graph for ' + a)
    ax1.plot(x,feature1)
    ax1.set_title('Feature 1 - MFCCs_10')
    ax2.plot(x,feature2)
    ax2.set_title('Feature 2 - MFCCs_17')
    ax1.set(xlabel='', ylabel='MFCCs')

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('line_graph_' + a + '.png')
    plt.show()

def boxPlot(data,x):
    """
    generate boxplot showing the distribution of features for both classes
    :param data:    raw data
    :param x:       filename
    :return:
    """
    # load features from the data
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # stores feature 1 values that belong to class 1
    f1category1 = []
    # stores feature 1 values that belong to class 2
    f1category2 = []
    # stores feature 2 values that belong to class 1
    f2category1 = []
    # stores feature 2 values that belong to class 2
    f2category2 = []

    # separate inputs based on its class
    category = data["Species"]
    for i in range(len(category)):
        s = category[i].decode("utf-8")
        if s == 'HylaMinuta':
            f1category1.append(feature1[i])
            f2category1.append(feature2[i])
        else:
            f1category2.append(feature1[i])
            f2category2.append(feature2[i])

    distribution = [f1category1,f1category2,f2category1,f2category2]
    fc = ['MFCCs_10 C1', 'MFCCs_10 C2', 'MFCCs_17 C1', 'MFCCs_17 C2']

    # generate boxplots for both classes for each feature
    fig, ax = plt.subplots()
    ax.boxplot(distribution)
    ax.set_xticklabels(fc)
    ax.set_title('BoxPlot for ' + x)

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('boxplot_' + x + '.png')
    plt.show()

def barGraph(data,x):
    """
    generate bar graph with error bars (For each class, 1 error bar per feature)
    :param data:    raw data
    :param x:       filename
    :return:
    """
    # load features from the data
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # stores feature 1 values that belong to class 1
    f1category1 = []
    # stores feature 1 values that belong to class 2
    f1category2 = []
    # stores feature 2 values that belong to class 1
    f2category1 = []
    # stores feature 2 values that belong to class 2
    f2category2 = []

    # separate inputs based on its class
    category = data["Species"]
    for i in range(len(category)):
        s = category[i].decode("utf-8")
        if s == 'HylaMinuta':
            f1category1.append(feature1[i])
            f2category1.append(feature2[i])
        else:
            f1category2.append(feature1[i])
            f2category2.append(feature2[i])

    # find mean of each class for each feature
    f1c1Mean = np.mean(f1category1)
    f1c2Mean = np.mean(f1category2)
    f2c1Mean = np.mean(f2category1)
    f2c2Mean = np.mean(f2category2)

    # find standard deviation of each class for each feature
    f1c1Std = np.std(f1category1)
    f1c2Std = np.std(f1category2)
    f2c1Std = np.std(f2category1)
    f2c2Std = np.std(f2category2)

    fc = ['MFCCs_10 C1', 'MFCCs_10 C2', 'MFCCs_17 C1','MFCCs_17 C2']
    x_pos = np.arange(len(fc))
    CTEs = [f1c1Mean, f1c2Mean, f2c1Mean, f2c2Mean]
    error = [f1c1Std, f1c2Std, f2c1Std, f2c2Std]

    # generate boxplots
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fc)
    ax.set_title('Bar Graph with Error Bars for '+x)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars'+x+'.png')
    plt.show()

def calculateStistics(data):
    """
    computes :
    1) the mean (expected value),
    2) covariance matrix, and
    3) standard deviation for each individual feature
    :param data:    raw data
    :return:
    """
    # load features from the data
    feature1 = data["MFCCs_10"]
    feature2 = data["MFCCs_17"]

    # mean of features
    feature1Mean = np.mean(feature1)
    feature2Mean = np.mean(feature2)

    # covariance matrix of features
    feature1Cov = np.cov(feature1)
    feature2Cov = np.cov(feature2)

    # standard deviation of features
    feature1Std = np.std(feature1)
    feature2Std = np.std(feature2)

    print("Feature 1: MFCCs_10")
    print("Mean (expected Value): ",feature1Mean)
    print("Covariance Matrix: ", feature1Cov)
    print("Standard Deviation: ", feature1Std)
    print()
    print("Feature 2: MFCCs_17")
    print("Mean (expected Value): ", feature2Mean)
    print("Covariance Matrix: ", feature2Cov)
    print("Standard Deviation: ", feature2Std)


if __name__ == "__main__":
    # load data from csv file
    sampledata = getData("Frogs-subsample.csv")
    data = getData("Frogs.csv")

    # Visualisation of data
    # display scatter plot
    scatterPlot(sampledata,'Frogs_subsample')
    scatterPlot(data,'Frogs')

    # display histogram
    histogramPlot(sampledata,'Frogs_subsample')
    histogramPlot(data,'Frogs')

    # display line graph
    lineGraph(sampledata,'Frogs_subsample')
    lineGraph(data,'Frogs')

    # display box plot
    boxPlot(sampledata,'Frogs_subsample')
    boxPlot(data,'Frogs')

    # display bar graph with error bar
    barGraph(sampledata,'Frogs_subsample')
    barGraph(data,'Frogs')

    # display statistics
    print('{:*^50}'.format("Statistics for Frogs-subsample.csv"))
    calculateStistics(sampledata)
    print()
    print('{:*^50}'.format("Statistics for Frogs.csv:"))
    calculateStistics(data)