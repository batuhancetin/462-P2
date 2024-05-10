import idx2numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers
import time


#need to install 
#pip3 install numpy idx2numpy --break-system-packages
#pip3 install scikit-learn --break-system-packages


def get_images_labels(required_digits):
    # file paths
    file_train_images = './mnist/train-images.idx3-ubyte'
    file_train_labels = './mnist/train-labels.idx1-ubyte'
    file_test_images = './mnist/t10k-images.idx3-ubyte'
    file_test_labels = './mnist/t10k-labels.idx1-ubyte'

    # loading the dataset
    train_images = idx2numpy.convert_from_file(file_train_images)
    train_labels = idx2numpy.convert_from_file(file_train_labels)
    test_images = idx2numpy.convert_from_file(file_test_images)
    test_labels = idx2numpy.convert_from_file(file_test_labels)

    # applying the filtering for required images
    train_images, train_labels = filter_digits(train_images, train_labels, required_digits)
    test_images, test_labels = filter_digits(test_images, test_labels, required_digits)

    return train_images, train_labels, test_images, test_labels

# it filters images w.r.t their labels
# it first only the labels consists required digits then return image and label list correspondingly
def filter_digits(images, labels, digits):
    mask = np.isin(labels, digits) # returns true false w.r.t labels
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    return filtered_images, filtered_labels

# it prints how many data exist in the set
def print_how_many_data(labels):
    dct = {}
    for i in labels:
        if i in dct.keys():
            dct[i] += 1
        else:
            dct[i] = 0
    
    print("Labels: " + str(dct))
    sum = 0
    for key in dct.keys():
        sum += dct[key]

    print("Sum: " + str(sum))

# it prints how many data exists in training and test sets
def print_test_train_labels(train_labels, test_labels):
    print_how_many_data(train_labels)
    print_how_many_data(test_labels)

# it will reshaping images from 28x28 matrix to 786-dimensional vector for svm
def reshaping_images_for_svm(train_images, test_images):
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    return train_images, test_images

# it normalizes the images, simplifies the calculations and execute faster
def normalize_images(train_images, test_images):
    return train_images / 255.0, test_images / 255.0

# it makes sets smaller to execute algorithms fast
def make_smaller_sets(train_images, test_images, train_labels, test_labels):
    train_images = train_images[:1000]
    test_images = test_images[:100]
    train_labels = train_labels[:1000]
    test_labels = test_labels[:100]
    return train_images, test_images, train_labels, test_labels

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

    return b,w

# it trains one vs all, returns each ones weights and bias to use it later
def train_ovr_svm(train_data, train_labels, required_digits):
    n_samples, n_features = train_data.shape
    models = []

    for label in required_digits:
        # Create binary labels for the current class
        binary_labels = np.where(train_labels == label, 1, -1)
        
        # Set up QP matrices
        Q, p, G, h = set_QP_matrices(train_data, binary_labels, n_samples, n_features)
        
        # Solve the QP problem for SVM parameters
        b, w = solve_QP(Q, p, G, h)
        models.append((w, b))
    
    return models

# it use trained model for testing data, return the predicted labels according to trained model
def predict_ovr_svm(models, data):

     # Compute the decision function from each SVM
     # x*w^T + b
    decision_values = [np.dot(data, model[0]) + model[1] for model in models]
    
    # Choose the highest indices for each model, in this way it merges all classifiers
    # The highest value describes the one in one vs all models
    predictions = np.argmax(np.array(decision_values), axis=0)

    # Map indices to original class labels, it converts 0,1,2,3 to 2,3,8,9 as the test label
    class_predictions = np.array(required_digits)[predictions]
    return class_predictions

# to stop printing its output
solvers.options['show_progress'] = False 


# required labels
required_digits = [2, 3, 8, 9]

# Load and preprocess data
train_images, train_labels, test_images, test_labels = get_images_labels(required_digits)
print_test_train_labels(train_labels, test_labels)
#train_images, test_images, train_labels, test_labels = make_smaller_sets(train_images, test_images, train_labels, test_labels)
train_images, test_images = reshaping_images_for_svm(train_images, test_images)
train_images,test_images = normalize_images(train_images,test_images)

start_time = time.time()
models = train_ovr_svm(train_images, train_labels, required_digits)
test_predictions = predict_ovr_svm(models, test_images)
end_time = time.time()
print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
print(f"Process completed in {end_time - start_time:.2f} seconds.")



