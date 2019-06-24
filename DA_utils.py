import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import colors
import matplotlib as mpl

# Initialization
bins = 30
alpha = 0.8
width = 15
height = 6
test_size = 30
num_sample_plot = 200


# Generate random variable sets with specified covariance
def multivariate_distribution(num_var, mean, cov_scale, num_datapoint, cov=None, plot=True):
    if cov == None:
        print('The covariance matrix is undefined! Program generates one by its own!')
        cov = cov_scale * np.random.rand(num_var, num_var)
        cov = np.dot(cov, cov.transpose())
    det = np.linalg.det(cov)
    print('covariance natrix:\n', cov)
    columns = []
    for i in np.arange(0, num_var, 1):
        columns = np.append(columns, 'var'+str(i+1))
    data = pd.DataFrame(data=np.random.multivariate_normal(mean, cov, num_datapoint), columns=columns)
    prob = []
    for i in np.arange(0, data.shape[0], 1):
        k = np.matmul(np.matmul(data.values[i, :] - mean, np.linalg.inv(cov)), (data.values[i, :]-mean)[:, np.newaxis])
        k = np.exp((-0.5) * k) / (np.sqrt(2*np.pi*det))
        prob = np.append(prob, k)
    data['joint_prob'] = prob
    if plot == True:
        data.hist(bins=bins, facecolor='b', alpha=alpha, figsize=(width, height))
        plt.suptitle('Distribution of random variables as well as their joint probability')
        plt.figure(figsize=(width, height))
        plt.title('Boxplot of generated variables')
        data.loc[:, data.columns != 'joint_prob'].boxplot(fontsize=12, grid=False)
    print('\nThe function returns a dictionary containing:\n'
          'data --> generated random variable sets\n'
          'covariance --> covariance matrix of generated data set\n'
          'det --> determinant of covariance matrix')
    return {'data': data, 'covariance': cov, 'det': det}


# Load and analyse iris data set
def load_analyse_iris(plot=False):
    print('\n***********************************************')
    print('Load and analyze the iris data set. This section is irrelevant to the one above.')

    # data prep
    X = load_iris()['data']
    y = load_iris()['target'][:, np.newaxis]
    data = pd.DataFrame(data=np.append(X, y, axis=1), columns=np.append(load_iris()['feature_names'], 'target'))
    features = list(data.columns[:-1])
    print('\nfeatures:', features)
    classes = np.unique(data['target'])
    print('classes:', classes)

    # Split features
    features_classes = {}
    if plot == True:
        plt.figure(figsize=(width, height))
        for i, ftr in enumerate(features):
            features_classes[ftr] = []
            for j, cls in enumerate(classes):
                features_classes[ftr].append(data.loc[data['target']==cls, data.columns[i]])
                plt.subplot('2'+str(int(np.ceil(len(features)/2)))+str(i+1))
                plt.suptitle('Distribution of classes vs features')
                plt.xlabel(ftr)
                plt.hist(features_classes[ftr], bins=bins, histtype='barstacked', alpha=alpha)

    # store most influential features
    data_filtered = data
    data_filtered = data_filtered.drop(['sepal length (cm)', 'sepal width (cm)'], axis=1)
    return data, data_filtered


def plot_clf(data, results):
    # data prep
    features = list(data.columns[:-1])
    classes = np.unique(data['target'])
    model = results['model']
    X = data.loc[:, data.columns != 'target'].values
    max = np.ceil(np.max(X, axis=0))
    min = np.floor(np.min(X, axis=0))
    xx, yy = np.meshgrid(np.linspace(min[0], max[0], num_sample_plot),
                             np.linspace(min[1], max[1], num_sample_plot))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # plot prep
    plt.figure(figsize=(width, height))
    features_classes = {}
    for j, cls in enumerate(classes):
        features_classes[cls] = []
        features_classes[cls] = data.loc[data['target'] == cls, data.columns != 'target'].values
        plt.subplot('1' + str(int(len(classes))) + str(j + 1))
        plt.suptitle('Classification results: '+results['description'])
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.xlim([min[0], max[0]])
        plt.ylim([min[1], max[1]])
        plt.pcolormesh(xx, yy, Z[:, j].reshape(xx.shape))
        plt.colorbar()
        plt.scatter(features_classes[cls][:, 0], features_classes[cls][:, 1], s=60, marker='o',
                    linewidths=1, edgecolors=[0, 0, 0], facecolor=[0.8, 0.8, 0.8], alpha=alpha)
    return 0


def linear_discriminant_analysis(data, solver='svd', shrinkage=True, tol=1e-4, store_covariance=True, plot=False):
    print('\n***********************************************')
    print('Linear Discriminant Analysis')
    print('\nAvailable solvers: svd, lsqr, eigen. Shrinkage can only be used with lsqr and eigen.')

    # data prep
    features = list(data.columns[:-1])
    print('\nfeatures:', features)
    classes = np.unique(data['target'])
    print('classes:', classes)
    X = data.loc[:, data.columns != 'target'].values
    y = data.loc[:, data.columns == 'target'].values
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=None, random_state=0)

    # lda model prep
    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, tol=tol, store_covariance=store_covariance)
    if solver != 'svd':
        model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, tol=tol, store_covariance=store_covariance)

    # lda model training
    model.fit(X_train, y_train)
    score_test = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    coef = model.coef_
    intercept = model.intercept_
    covariance = model.covariance_
    means = model.means_
    results = {'description': 'Linear Discriminant Analysis', 'model': model, 'score_test': score_test,
            'score_train': score_train, 'coef': coef, 'intercept': intercept, 'covariance': covariance, 'means': means}

    # Plot results
    if plot == True:
        if len(features) == 2:
            plot_clf(data, results)
        else:
            print('Plot does not work for the number of features larger than two.')
    return results


def quadratic_discriminant_analysis(data, reg_param=0.0, tol=1e-4, store_covariance=True, plot=False):
    print('\n***********************************************')
    print('Quadratic Discriminant Analysis')

    # data prep
    features = list(data.columns[:-1])
    print('\nfeatures:', features)
    classes = np.unique(data['target'])
    print('classes:', classes)
    X = data.loc[:, data.columns != 'target'].values
    y = data.loc[:, data.columns == 'target'].values
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=None, random_state=0)

    # lda model prep
    model = QuadraticDiscriminantAnalysis(reg_param=reg_param, tol=tol, store_covariance=store_covariance)

    # lda model training
    model.fit(X_train, y_train)
    score_test = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    covariance = model.covariance_
    means = model.means_
    results = {'description': 'Linear Discriminant Analysis', 'model': model, 'score_test': score_test,
            'score_train': score_train, 'covariance': covariance, 'means': means}

    # Plot results
    if plot == True:
        if len(features) == 2:
            plot_clf(data, results)
        else:
            print('Plot does not work for the number of features larger than two.')
    return results


