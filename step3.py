import idx2numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers
from sklearn.decomposition import PCA
import time

#need to install 
#pip3 install numpy idx2numpy --break-system-packages
#pip3 install scikit-learn --break-system-packages
solvers.options['show_progress'] = True  # For more verbose output


def get_images_labels():
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

    # required labels
    required_digits = [2, 3, 8, 9]

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

# it makes sets smaller to execute algorithms fast
def make_smaller_sets(train_images, test_images, train_labels, test_labels):
    train_images = train_images[:1000]
    test_images = test_images[:100]
    train_labels = train_labels[:1000]
    test_labels = test_labels[:100]
    return train_images, test_images, train_labels, test_labels

# it normalizes the images, simplifies the calculations and execute faster
def normalize_images(train_images, test_images):
    return train_images / 255.0, test_images / 255.0

# define polynomial kernel 
def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x-y)**2)

# set up qp matrix
def set_qp_matrix(train_data):
    n_samples = train_data.shape[0]
    QP = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            QP[i,j] = rbf_kernel(train_data[i], train_data[j])
    return QP


# it arranges matrices for solving Quadratic Programming
def arrange_matrices(train_data, train_labels, C):
    n_samples = train_data.shape[0]
    QP = set_qp_matrix(train_data)

    # it makes calculation easy, basically y=[1,-1] -> y_outer = [[1,-1][-1,1]]
    y_outer = np.outer(train_labels, train_labels)
    # Pij = yi*yj*K(xi,xj) therefore we need to 
    P = matrix(y_outer * QP)
    #  column vector of minus ones for maximizing ∑α(i) which means minimize -∑α(i)
    q = matrix(-np.ones(n_samples))
    # identity matrix for making C>=a(i)>=0
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    # zeros for making a(i)>=0, C values for making C>=a(i), where C is the regularization parameter
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = matrix(train_labels.reshape(1, -1), tc='d')
    # scalar 0 for ∑α(i)y(i) = 0 's right handside
    b = matrix(0.0)
    return P, q, G, h, A, b

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
        #print(sum_part)
        b_k = y_k - sum_part
        b_values.append(b_k)
    # if it's empty somehow, it returns 0
    if b_values:
        return np.mean(b_values)
    else:
        return 0 

# it trains one versus all svm model w.r.t training samples
def train_ovr_svm(train_data, train_labels, required_digits, regularization_param=1):
    models = []
    for digit in required_digits:
        # binary labels for the current class vs all others
        binary_labels = np.where(train_labels == digit, 1, -1)
        # set QP matrices using the dual formulation, it is updated w.r.t binary_labels
        P, q, G, h, A, b = arrange_matrices(train_data, binary_labels, regularization_param)
        # get alpha values according to QP matrices
        alphas = get_alphas(P, q, G, h, A, b)
        # find support vectors and with this information, calculate bias term
        support_vector_indices, support_vectors, support_vector_labels, support_vector_alphas = get_support_vectors(alphas, train_data, binary_labels)
        bias = calculate_bias(alphas, binary_labels, set_qp_matrix(train_data), support_vector_labels, support_vector_indices)
        models.append((alphas, bias))
    
    return models

# function that predicts the test data according to trained model
def predict_ovr_svm(models, test_data, train_data, train_labels, required_digits):
    num_tests = test_data.shape[0]
    num_classes = len(models)
    decision_values = np.zeros((num_tests, num_classes))

    # iterating for each classifier
    for idx, (alphas, bias) in enumerate(models):
        # calculating each decision value for each test sample
        for i in range(num_tests):
            # prediction is f(x)=sign(⟨w,ϕ(x)⟩+b) so convert w into ∑α(i)y(i)ϕ(x(i)) and that makes kernel
            # decision_value = sign(∑α(i)y(i)K(x(j),x(i)) + b)
            decision_value = sum(alphas[j] * train_labels[j] * rbf_kernel(train_data[j], test_data[i]) for j in range(len(train_data))) + bias
            decision_values[i, idx] = decision_value

    # assign labels w.r.t the highest value according to classifiers
    predictions_indices = np.argmax(decision_values, axis=1) 
    class_predictions = np.array(required_digits)[predictions_indices]
    return class_predictions

# this function extracts the features, it defaults extract features until 50 features remaining (from 786 28*28)
def apply_pca(train_images, test_images, n_components=50):
    start_time = time.time()
    pca = PCA(n_components=n_components)
    pca.fit(train_images)
    train_images_pca = pca.transform(train_images)
    test_images_pca = pca.transform(test_images)
    end_time = time.time()
    print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
    return train_images_pca, test_images_pca

# load data
train_images, train_labels, test_images, test_labels = get_images_labels()
#train_images, test_images, train_labels, test_labels = make_smaller_sets(train_images, test_images, train_labels, test_labels)
print_test_train_labels(train_labels, test_labels)
train_images, test_images = reshaping_images_for_svm(train_images, test_images)
train_images,test_images = normalize_images(train_images,test_images)
train_labels = np.array(train_labels, dtype=np.double)
test_labels = np.array(test_labels, dtype=np.double)


required_digits = [2, 3, 8, 9]
train_images, test_images = apply_pca(train_images, test_images)
# start the model
start_time = time.time()
models = train_ovr_svm(train_images, train_labels, required_digits)
predicted_labels = predict_ovr_svm(models, test_images, train_images, train_labels, required_digits)
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy on test data:", accuracy)
end_time = time.time()
print(f"Process completed in {end_time - start_time:.2f} seconds.")
