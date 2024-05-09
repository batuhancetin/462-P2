import numpy as np
from cvxopt import matrix, solvers

# need to install pip install cvxopt

# Example Data (X) and Labels (y)
X = np.array([[0, 0],[2, 2],[2, 0],[3, 0]])
y = np.array([-1, -1, 1, 1])
print(y)

print(X.shape)

# Number of samples
n_samples, n_features = X.shape #4,2


# Set up QP matrices
Q = np.zeros((n_features + 1, n_features + 1)) # create a zero matrix at first, to complete 0d and 0d^T easily in initialization
Q[1:, 1:] = np.eye(n_features)  # identity matrix in lower right

p = np.zeros(n_features + 1)

# Constructing the matrix A for inequalities
# y * (w^T * x + b) >= 1 for each sample
A = np.zeros((n_samples, n_features + 1)) #4,3

A[:, 0] = y  # the first column of G to be equal to y
y_col = y[:, np.newaxis] # convert y into a column vector for multipliyng with X and get the rest
A[:, 1:] = y_col * X  #for each row the rest of the indices should = y_i * x_i^T which is y_col * X
print(A)


c = np.ones(n_samples)  # The >= 1 vector

# Convert to cvxopt format
Q = matrix(Q)
p = matrix(p)

# In quadratic solver the equation is  Gx<=h but we need to convert it Au>=c, therefore we multiply them with -1
G = matrix(-A)  # Inequality needs to be in the form of Gx <= h
h = matrix(-c)

# Solve QP problem 
solution = solvers.qp(Q, p, G, h)


# Extract weights and bias
weights = np.array(solution['x']).flatten() #make it row vector instead of column
b = weights[0] # first one is bias
w = weights[1:] # rest is the weights

print("Bias (b):", b)
print("Weights (w):", w)


