import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# need to install pip install cvxopt

#it initializes data
def init_data():
    # Example Data (X) and Labels (y)
    train_data = np.array([[0, 0],[2, 2],[2, 0],[3, 0]])
    train_labels = np.array([-1, -1, 1, 1])
    # Number of samples
    n_samples, n_features = train_data.shape #4,2    
    return train_data, train_labels, n_samples, n_features

# it prepares QP matrices for solving the quadratic equation
def set_QP_matrices(train_data, train_labels, n_samples, n_features):
    # Set up QP matrices
    Q = np.zeros((n_features + 1, n_features + 1)) # create a zero matrix at first, to complete 0d and 0d^T easily in initialization
    Q[1:, 1:] = np.eye(n_features)  # identity matrix in lower right

    p = np.zeros(n_features + 1)

    # Constructing the matrix A for inequalities
    # y * (w^T * x + b) >= 1 for each sample
    A = np.zeros((n_samples, n_features + 1)) #4,3

    A[:, 0] = train_labels  # the first column of G to be equal to y
    y_col = train_labels[:, np.newaxis] # convert y into a column vector for multipliyng with X and get the rest
    A[:, 1:] = y_col * train_data  #for each row the rest of the indices should = y_i * x_i^T which is y_col * X

    c = np.ones(n_samples)  # The >= 1 vector

    # Convert to cvxopt format
    Q = matrix(Q)
    p = matrix(p)

    # In quadratic solver the equation is  Gx<=h but we need to convert it Au>=c, therefore we multiply them with -1
    G = matrix(-A)  # Inequality needs to be in the form of Gx <= h
    h = matrix(-c)
    return Q,p,G,h

# it solve QP with given parameters and return bias and weights
def solve_QP(Q,p,G,h):
    # Solve QP problem 
    solution = solvers.qp(Q, p, G, h)


    # Extract weights and bias
    weights = np.array(solution['x']).flatten() #make it row vector instead of column
    b = weights[0] # first one is bias
    w = weights[1:] # rest is the weights

    print("Bias (b):", b)
    print("Weights (w):", w)
    return b,w

# it calculates the support vectors
def calc_support_vectors(w, train_data, train_labels):
    # Calculate the norm of w to calculate the margin
    norm_w = np.linalg.norm(w)
    # Calculate the margin
    margin = 2 / norm_w
    print("Margin:", margin)

    # Calculate distances from the hyperplane
    distances = train_labels * (np.dot(train_data, w) + b)

    # Identify support vectors
    support_vectors = train_data[np.isclose(distances, 1)]
    print("Support Vectors:\n", support_vectors)
    return support_vectors

# it plots data and show support vectors by circling them
def plot_data(train_data, train_labels, w, b, support_vectors):
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(-1, 5, 50), np.linspace(-1, 5, 50))

    # Calculate decision boundary
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=plt.cm.bwr)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.show()

train_data, train_labels, n_samples, n_features = init_data()
Q,p,G,h = set_QP_matrices(train_data, train_labels, n_samples, n_features)
b,w = solve_QP(Q,p,G,h)
support_vectors = calc_support_vectors(w, train_data, train_labels)
plot_data(train_data, train_labels, w, b, support_vectors)

