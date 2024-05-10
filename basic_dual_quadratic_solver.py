import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# generate synthetic data
def generate_data():
    np.random.seed(0)
    train_data = np.vstack(((np.random.randn(10, 2) + [2, 2]), (np.random.randn(10, 2) - [2, 2])))
    train_labels = np.hstack((np.ones(10), -1*np.ones(10)))
    return train_data,train_labels

# define polynomial kernel 
def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

# set up qp matrix
def set_qp_matrix(train_data):
    n_samples, n_features = train_data.shape
    QP = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            QP[i,j] = polynomial_kernel(train_data[i], train_data[j])
    return QP

# it arranges matrices for solving Quadratic Programming
def arrange_matrices(train_data, train_labels):
    QP = set_qp_matrix(train_data)
    n_samples, n_features = train_data.shape
    # it makes calculation easy, basically y=[1,-1] -> y_outer = [[1,-1][-1,1]]
    y_outer = np.outer(train_labels, train_labels)
    # Pij = yi*yj*K(xi,xj) therefore we need to 
    P = matrix(y_outer * QP)
    #  column vector of minus ones maximize ∑α(i) means minimize -∑α(i)
    q = matrix(-np.ones((n_samples, 1)))
    # identity matrix
    G = matrix(-np.eye(n_samples))
    # zero matrix as length of samples
    h = matrix(np.zeros(n_samples))
    # row vector with labels
    A = matrix(train_labels, (1, n_samples), 'd')
    # scalar 0 for ∑α(i)y(i) = 0 's right handside
    b = matrix(0.0)
    return P,q,G,h,A,b

# get alpha values for each data by solving qp problem
def get_alphas(P,q,G,h,A,b):
    # solving qp problem
    solution = solvers.qp(P, q, G, h, A, b)
    # lagrange multipliers for each data point in the sample α(i)
    alphas = np.array(solution['x']).flatten()
    return alphas

# Get the support vectors
# Here was the equation: y(i)(w^Tx(i)+b) > 1-α(i) when a near to 0, it means that is not violating the soft margin, so it's a correctly classified
# when α(i) bigger than some threshold, they would be the exception point which is support vector so as a result
# if their alphas greater than threshold, they can be support vectors
def get_support_vectors(alphas, train_data, train_labels):
    support_vector_indices = np.where(alphas > 1e-5)[0]
    support_vectors = train_data[support_vector_indices]
    support_vector_labels = train_labels[support_vector_indices]
    support_vector_alphas = alphas[support_vector_indices]
    return support_vector_indices,support_vectors, support_vector_labels, support_vector_alphas

# calculate the bias term from y(i)(w^Tx(i) + b) = 1 (for support vectors for sure)
# substitute w as ∑α(i)y(i)x(i) and convert x's into ϕ(x)s what we made in kernel
# So, b = y(i) - ∑α(i)y(i)K(x(j),x(i)) take the mean value at the end
def calculate_bias(alphas, train_labels, QP, support_vector_labels, support_vector_indices):
    b_values = []
    for y_k, ind_support_vector in zip(support_vector_labels, support_vector_indices):
        # Calculate the sum part of the equation for each support vector
        # ∑α(i)y(i)
        sum_part = np.sum(alphas * train_labels * QP[ind_support_vector, :])
        # Calculate the intercept for each support vector and store it
        print(sum_part)
        b_k = y_k - sum_part
        b_values.append(b_k)

    # Average all calculated b values to get the final intercept
    return np.mean(b_values)

# prediction is f(x)=sign(⟨w,ϕ(x)⟩+b) so convert w into ∑α(i)y(i)ϕ(x(i)) and that makes kernel
# sign(∑α(i)y(i)K(x(j),x(i)) + b)
# Function to make predictions for each data sample
def predict(test_data, train_data, train_labels, alphas, b):
    predictions = []
    # Iterate over each data point to be classified
    for X_new in test_data:
        # Calculate kernelized product sum for the new data point with all support vectors
        kernelized_sum = np.sum(alphas * train_labels * np.array([polynomial_kernel(x_i, X_new) for x_i in train_data]))
        decision_value = kernelized_sum + b
        predicted_class = np.sign(decision_value)
        predictions.append(predicted_class)
    return np.array(predictions)


train_data,train_labels = generate_data()
QP = set_qp_matrix(train_data)
P,q,G,h,A,b = arrange_matrices(train_data, train_labels)
alphas= get_alphas(P,q,G,h,A,b)
support_vector_indices,support_vectors, support_vector_labels, support_vector_alphas = get_support_vectors(alphas, train_data, train_labels)
b = calculate_bias(alphas, train_labels, QP, support_vector_labels, support_vector_indices)
prediction_labels = predict(train_data, train_data, train_labels, alphas, b)
accuracy = accuracy_score(train_labels, prediction_labels)
print("Accuracy:", accuracy)

# Plotting the data and support vectors
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=plt.cm.Paired, s=30)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('SVM with Polynomial Kernel')
plt.show()
