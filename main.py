from DA_utils import *
import matplotlib.pyplot as plt


# Initialization
num_var = 5
num_datapoint = 500
mean = np.zeros(num_var)
cov_scale = 10
cov = [[1, 0], [0, 1]]
solver_lda = 'svd'
shrinkage = 0.5
tol = 1e-8
store_covariance = True
reg_param = 0.0

# Generate random variable sets with specified covariance matrix
results_multivar = multivariate_distribution(num_var=num_var, mean=mean, cov_scale=cov_scale, num_datapoint=num_datapoint,
                                    plot=False)

# Load and analyze iris dataset
data, data_filtered = load_analyse_iris(plot=False)

# Linear Discriminant Analysis
results_lda = linear_discriminant_analysis(data_filtered, solver=solver_lda, shrinkage=shrinkage, tol=tol,
                             store_covariance=store_covariance, plot=False)
print('Train score = ', results_lda['score_train'])
print('Test score = ', results_lda['score_test'])
print('Fixed covariance matrix of classes =\n', results_lda['covariance'])

# Quadratic Discriminant Analysis
results_qda = quadratic_discriminant_analysis(data_filtered, reg_param=reg_param, tol=tol, store_covariance=True, plot=False)
print('Train score = ', results_qda['score_train'])
print('Test score = ', results_qda['score_test'])
print('covariance matrices of classes =\n', results_qda['covariance'])

# Naive Bayes Analysis
results_qda = gaussian_naive_bayes_analysis(data_filtered, plot=True)
print('Train score = ', results_qda['score_train'])
print('Test score = ', results_qda['score_test'])


plt.show()

