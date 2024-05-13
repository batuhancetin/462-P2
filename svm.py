import idx2numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#need to install 
#pip3 install tensorflow --break-system-packages
#pip3 install numpy idx2numpy --break-system-packages
#pip3 install scikit-learn --break-system-packages
#pip3 install scikit-image --break-system-packages

# These are common steps for every run
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
    print("Min and Max of train_images  in normalization :", np.min(train_images / 255.0), np.max(train_images / 255.0))
    print("Min and Max of train_images in normalization :", np.min(test_images / 255.0), np.max(test_images / 255.0))
    return train_images / 255.0, test_images / 255.0


# it makes sets smaller to execute algorithms fast
def make_smaller_sets(train_images, test_images, train_labels, test_labels):
    train_images = train_images[:1000]
    test_images = test_images[:100]
    train_labels = train_labels[:1000]
    test_labels = test_labels[:100]
    return train_images, test_images, train_labels, test_labels

# it arranges solver's settings
def arrange_cvxopt_solver():
    solvers.options['show_progress'] = False 
    solvers.options['feastol'] = 1e-9
    solvers.options['abstol'] = 1e-9
    solvers.options['reltol'] = 1e-9
    solvers.options['maxiters'] = 200

class LinearQuadraticSolver:

    def __init__(self, train_images, test_images, train_labels, test_labels, regularization_param = 1e-6, is_feature_extract=False):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.regularization_param = regularization_param
        self.is_feature_extract = is_feature_extract
        self.required_digits = [2, 3, 8, 9]
        self.step1()

    def step1(self):
        if (self.is_feature_extract):
            self.train_images, self.test_images = self.apply_pca(self.train_images, self.test_images, 48)
            
        n_samples = self.train_images.shape[0]
        start_time = time.time()
        models = self.train_ovr_svm(self.train_images, self.train_labels, self.required_digits, n_samples)
        test_predictions = self.predict_ovr_svm(models, self.test_images)
        end_time = time.time()
        print("Test Accuracy:", accuracy_score(self.test_labels, test_predictions))
        print(f"Process completed in {end_time - start_time:.2f} seconds.")


    # it prepares QP matrices for solving the quadratic equation
    def set_QP_matrices(self, train_data, train_labels, n_samples):

        n_features = train_data.shape[1]  # Update to use the correct number of features after PCA
        # Set up QP matrices
        Q = np.zeros((n_features + 1, n_features + 1)) # create a zero matrix at first, to complete 0d and 0d^T easily in initialization
        #Q[1:, 1:] = np.eye(n_features)  # identity matrix in lower right
        Q[1:, 1:] = np.eye(n_features) * (1 + self.regularization_param)
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
        
        # Debug print
        eigenvalues = np.linalg.eigvals(Q)
        if np.any(eigenvalues < 0):
            print("Q is not positive semidefinite.")
        print(f"Q shape: {Q.size}, p shape: {p.size}, G shape: {G.size}, h shape: {h.size}")
        #print("Check Q matrix for negative diagonal elements:", np.diag(Q))
        print("Eigenvalues of Q:", np.linalg.eigvals(Q))
        print("Check min/max of G:", np.min(G), np.max(G))
        print("Check min/max of h:", np.min(h), np.max(h))
        
        print(f"Q shape: {Q.size}, p shape: {p.size}, G shape: {G.size}, h shape: {h.size}")
        return Q,p,G,h


    # it solve QP with given parameters and return bias and weights
    def solve_QP(self, Q,p,G,h):
        # Attempt to solve QP problem 
        
        solution = solvers.qp(Q, p, G, h)
        if solution['status'] != 'optimal':
            print("Warning: Not optimal solution found.")
        #solution = solvers.qp(Q, p, G, h)
        print("Solution: " + str(solution))        
        # Extract weights and bias
        weights = np.array(solution['x']).flatten() #make it row vector instead of column
        b = weights[0] # first one is bias
        w = weights[1:] # rest is the weights
        return b, w

    # it trains one vs all, returns each ones weights and bias to use it later
    def train_ovr_svm(self, train_data, train_labels, required_digits, n_samples):
        models = []

        for label in required_digits:
            #creating binary labels for the current class
            binary_labels = np.where(train_labels == label, 1, -1)
            
            # set QP matrices for each class
            Q, p, G, h = self.set_QP_matrices(train_data, binary_labels, n_samples)
            
            # solving the QP problem for SVM parameters
            b, w = self.solve_QP(Q, p, G, h)
            models.append((w, b))
        
        return models

    # it use trained model for testing data, return the predicted labels according to trained model
    def predict_ovr_svm(self, models, data):

        # Compute the decision function from each SVM
        # x*w^T + b
        decision_values = [np.dot(data, model[0]) + model[1] for model in models]

        # Choose the highest indices for each model, in this way it merges all classifiers
        # The highest value describes the one in one vs all models
        predictions = np.argmax(np.array(decision_values), axis=0)

        # Map indices to original class labels, it converts 0,1,2,3 to 2,3,8,9 as the test label
        class_predictions = np.array(required_digits)[predictions]
        return class_predictions

    # it applies pca to test and train images
    def apply_pca(self, train_images, test_images, n_components=50):
        train_images,test_images = self.normalize_images_before_pca(train_images,test_images)
        start_time = time.time()
        pca = PCA(n_components=n_components)
        pca.fit(train_images)
        train_images_pca = pca.transform(train_images)
        test_images_pca = pca.transform(test_images)
        end_time = time.time()
        print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
        #train_images, test_images = self.normalize_pca_output(train_images, test_images)
        return train_images_pca, test_images_pca

    # it normalizes the images, simplifies the calculations and execute faster
    # to prevent error in pca, we need to update normalizer from basic return train_images / 255.0, test_images / 255.0
    def normalize_images_before_pca(self,train_images,test_images):
        # Initialize the scaler
        train_images,test_images = train_images * 255.0,test_images *255.0
        scaler = StandardScaler()

        # Fit on the training data and transform both train and test data
        train_images_scaled = scaler.fit_transform(train_images)
        test_images_scaled = scaler.fit_transform(test_images)
        print("Min and Max of train_images before pca:", np.min(train_images_scaled), np.max(train_images_scaled))
        print("Min and Max of train_images before pca:", np.min(test_images_scaled), np.max(test_images_scaled))
        return train_images_scaled, test_images_scaled


    # normalizing pca outputs to not raise an error
    def normalize_pca_output(self, train_images_pca, test_images_pca):
        # Find global minimum and maximum
        min_val = np.min(train_images_pca)
        max_val = np.max(train_images_pca)
        print("Min and Max of train_images after PCA:", min_val, max_val)
        min_val_test = np.min(test_images_pca)
        max_val_test = np.max(test_images_pca)
        print("Min and Max of test_images after PCA:", min_val_test, max_val_test)
        # Scale both training and testing data using the training set's min and max
        train_images_normalized = (train_images_pca - min_val) / (max_val - min_val)
        test_images_normalized = (test_images_pca - min_val_test) / (max_val_test - min_val_test) 
        print("Min and Max of train_images after normalization:", np.min(train_images_normalized), np.max(train_images_normalized))
        print("Min and Max of test_images after normalization:", np.min(test_images_normalized), np.max(test_images_normalized))
        return train_images_normalized, test_images_normalized

class ScikitLinearKernel:

    def __init__(self, train_images, test_images, train_labels, test_labels, regularazation_param, is_feature_extract=True, find_best_params=False):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.is_feature_extract = is_feature_extract
        self.regularazation_param = regularazation_param
        self.find_best_params = find_best_params
        self.required_digits = [2, 3, 8, 9]
        self.step_b(train_images, train_labels, test_images, test_labels, is_feature_extract, find_best_params)


    # it trains scikit-learn’s soft margin primal SVM function with linear kernel. Its default regularization param is 1.0 to make it soft-margin
    # if is_feature_extraction is true, it first extracts the features
    def step_b(self, train_images, train_labels, test_images, test_labels, is_feature_extraction, find_best_params):
        regulirazation_param = self.regularazation_param
        if (is_feature_extraction):
            train_images, test_images = self.apply_pca(train_images, test_images)
        self.start_svm(regulirazation_param, train_images, train_labels, test_images, test_labels)
        if (find_best_params):
            self.find_best_param(train_images, train_labels, test_images, test_labels)

    # it starts linar svm with given regularization parameters and image & label datasets 
    def start_svm(self, regularazation_param, train_images, train_labels, test_images, test_labels):
        print("Regularization Param (C): " + str(regularazation_param))
        start_time = time.time()
        # Initialize the LinearSVC model with default parameters to start
        # dual=False when n_samples > n_features
        svm_model = LinearSVC(C=regularazation_param,dual=False)  
        svm_model.fit(train_images, train_labels)

        train_predictions = svm_model.predict(train_images)
        test_predictions = svm_model.predict(test_images)

        print("Training Accuracy: ", accuracy_score(train_labels, train_predictions))
        print("Testing Accuracy: ", accuracy_score(test_labels, test_predictions))
        print("\nClassification Report:\n", classification_report(test_labels, test_predictions))
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")

    # Warning: It takes to much time
    # It will finds the best regularization parameter for svm
    def find_best_param(self, train_images, train_labels, test_images, test_labels):
        # differnt C parameters
        # A higher value of C allows for fewer margin violations (hard margin), while a lower value of C allows for more margin violations (soft margin)
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

        start_time = time.time()

        # Grid search with cross-validation
        #n_jobs for parallel programming it tells scikit-learn to use all available CPU cores to perform the operations.
        grid_search = GridSearchCV(LinearSVC(dual=False), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(train_images, train_labels)

        print("Best Parameters: ", grid_search.best_params_)
        print("Best Cross-validation Accuracy: ", grid_search.best_score_)

        # Evaluate using the best model found by GridSearchCV
        best_model = grid_search.best_estimator_
        best_predictions = best_model.predict(test_images)
        print("Test Accuracy with Best Model: ", accuracy_score(test_labels, best_predictions))
        print("\nBest Model Classification Report:\n", classification_report(test_labels, best_predictions))

        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")

    # this function extracts the features, it defaults extract features until 50 features remaining (from 786 28*28)
    def apply_pca(self, train_images, test_images, n_components=50):
        start_time = time.time()
        pca = PCA(n_components=n_components)
        pca.fit(train_images)
        train_images_pca = pca.transform(train_images)
        test_images_pca = pca.transform(test_images)
        end_time = time.time()
        print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
        return train_images_pca, test_images_pca

class NonLinearQuadraticSolver():

    def __init__(self, train_images, test_images, train_labels, test_labels, is_feature_extract=True):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.is_feature_extract = is_feature_extract
        self.required_digits = [2, 3, 8, 9]
        self.step_c(train_images, train_labels, test_images, test_labels, is_feature_extract)

    def step_c(self, train_images, train_labels, test_images, test_labels, is_feature_extract):
        train_labels = np.array(train_labels, dtype=np.double)
        test_labels = np.array(test_labels, dtype=np.double)
        
        if (is_feature_extract):
            train_images, test_images = self.apply_pca(train_images, test_images)
        # start the model
        start_time = time.time()
        models = self.train_ovr_svm(train_images, train_labels, required_digits)
        predicted_labels = self.predict_ovr_svm(models, test_images, train_images, train_labels, required_digits)
        accuracy = accuracy_score(test_labels, predicted_labels)
        print("Accuracy on test data:", accuracy)
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")

    # define polynomial kernel 
    def polynomial_kernel(self, x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def rbf_kernel(self, x, y, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x-y)**2)

    # set up qp matrix
    def set_qp_matrix(self,train_data):
        n_samples = train_data.shape[0]
        QP = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                QP[i,j] = self.rbf_kernel(train_data[i], train_data[j])
        return QP


    # it arranges matrices for solving Quadratic Programming
    def arrange_matrices(self, train_data, train_labels, C):
        n_samples = train_data.shape[0]
        QP = self.set_qp_matrix(train_data)

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
    def get_alphas(self, P,q,G,h,A,b):
        # solving qp problem
        solution = solvers.qp(P, q, G, h, A, b)
        # lagrange multipliers for each data point in the sample α(i)
        alphas = np.array(solution['x']).flatten()
        return alphas

    # Get the support vectors
    # Here was the equation: y(i)(w^Tx(i)+b) > 1-α(i) when a near to 0, it means that is not violating the soft margin, so it's a correctly classified
    # when α(i) bigger than some threshold, they would be the exception point which is support vector so as a result
    # if their alphas greater than threshold, they can be support vectors
    def get_support_vectors(self, alphas, train_data, train_labels):
        support_vector_indices = np.where(alphas > 1e-5)[0]
        support_vectors = train_data[support_vector_indices]
        support_vector_labels = train_labels[support_vector_indices]
        support_vector_alphas = alphas[support_vector_indices]
        return support_vector_indices,support_vectors, support_vector_labels, support_vector_alphas

    # calculate the bias term from y(i)(w^Tx(i) + b) = 1 (for support vectors for sure)
    # substitute w as ∑α(i)y(i)x(i) and convert x's into ϕ(x)s what we made in kernel
    # So, b = y(i) - ∑α(i)y(i)K(x(j),x(i)) take the mean value at the end
    def calculate_bias(self, alphas, train_labels, QP, support_vector_labels, support_vector_indices):
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
    def train_ovr_svm(self, train_data, train_labels, required_digits, regularization_param=1):
        models = []
        for digit in required_digits:
            # binary labels for the current class vs all others
            binary_labels = np.where(train_labels == digit, 1, -1)
            # set QP matrices using the dual formulation, it is updated w.r.t binary_labels
            P, q, G, h, A, b = self.arrange_matrices(train_data, binary_labels, regularization_param)
            # get alpha values according to QP matrices
            alphas = self.get_alphas(P, q, G, h, A, b)
            # find support vectors and with this information, calculate bias term
            support_vector_indices, support_vectors, support_vector_labels, support_vector_alphas = self.get_support_vectors(alphas, train_data, binary_labels)
            bias = self.calculate_bias(alphas, binary_labels, self.set_qp_matrix(train_data), support_vector_labels, support_vector_indices)
            models.append((alphas, bias))
        
        return models

    # function that predicts the test data according to trained model
    def predict_ovr_svm(self, models, test_data, train_data, train_labels, required_digits):
        num_tests = test_data.shape[0]
        num_classes = len(models)
        decision_values = np.zeros((num_tests, num_classes))

        # iterating for each classifier
        for idx, (alphas, bias) in enumerate(models):
            # calculating each decision value for each test sample
            for i in range(num_tests):
                # prediction is f(x)=sign(⟨w,ϕ(x)⟩+b) so convert w into ∑α(i)y(i)ϕ(x(i)) and that makes kernel
                # decision_value = sign(∑α(i)y(i)K(x(j),x(i)) + b)
                decision_value = sum(alphas[j] * train_labels[j] * self.rbf_kernel(train_data[j], test_data[i]) for j in range(len(train_data))) + bias
                decision_values[i, idx] = decision_value

        # assign labels w.r.t the highest value according to classifiers
        predictions_indices = np.argmax(decision_values, axis=1) 
        class_predictions = np.array(required_digits)[predictions_indices]
        return class_predictions

    # this function extracts the features, it defaults extract features until 50 features remaining (from 786 28*28)
    def apply_pca(self, train_images, test_images, n_components=50):
        start_time = time.time()
        pca = PCA(n_components=n_components)
        pca.fit(train_images)
        train_images_pca = pca.transform(train_images)
        test_images_pca = pca.transform(test_images)
        end_time = time.time()
        print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
        return train_images_pca, test_images_pca

class ScikitNonLinearKernel:

    def __init__(self, train_images, test_images, train_labels, test_labels, is_feature_extract=True, display_support_vector=False, find_best_params = False):
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.is_feature_extract = is_feature_extract
        self.display_support_vector = display_support_vector
        self.find_best_params = find_best_params
        self.required_digits = [2, 3, 8, 9]
        self.step_d(train_images, train_labels, test_images, test_labels, is_feature_extract, display_support_vector, find_best_params)

    def step_d(self, train_images, train_labels, test_images, test_labels, is_feature_extract, display_support_vectors, find_best_params):
        if (is_feature_extract):
            train_images, test_images = self.apply_pca(train_images, test_images)

        start_time = time.time()
        svm_classifier = self.train_and_evaluate_svm(train_images, train_labels, test_images, test_labels)
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")
        if (find_best_params):
            #self.find_best_hyperparam(train_images, train_labels)
            #self.find_best_kernel(train_images, train_labels)
            self.find_best_gamma(train_images, train_labels)
        # if pca is applied, we cannot display
        if display_support_vectors and (not is_feature_extract):
            self.display_support_vectors(svm_classifier,train_images,train_labels)

    def train_and_evaluate_svm(self, train_images, train_labels, test_images, test_labels):
        # arranging SVM classifier with a non-linear (rbf) kernel
        svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42)

        # train the model
        svm_classifier.fit(train_images, train_labels)

        # predicting the test set results
        y_pred = svm_classifier.predict(test_images)

        # evaluating the model by comparing prediction and actual labels
        accuracy = accuracy_score(test_labels, y_pred)
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(test_labels, y_pred))

        return svm_classifier

    # this function extracts the features, it defaults extract features until 50 features remaining (from 786 28*28)
    def apply_pca(self, train_images, test_images, n_components=50):
        start_time = time.time()
        pca = PCA(n_components=n_components)
        pca.fit(train_images)
        train_images_pca = pca.transform(train_images)
        test_images_pca = pca.transform(test_images)
        end_time = time.time()
        print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
        return train_images_pca, test_images_pca


    # finding the best C
    def find_best_hyperparam(self, train_images, train_labels): 
        start_time = time.time()
        parameters = {'kernel':('poly', 'rbf'), 'C':[0.1, 1,10]}
        grid_search = GridSearchCV(SVC(kernel='rbf'), parameters)
        grid_search.fit(train_images, train_labels)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        end_time = time.time()
        print(f"find_best_hyperparam completed in {end_time - start_time:.2f} seconds.")

    # finding the best kernel method
    def find_best_kernel(self, train_images, train_labels): 
        start_time = time.time()
        parameters = {'kernel':('poly', 'rbf', 'sigmoid')}
        grid_search = GridSearchCV(SVC(C=10), parameters)
        grid_search.fit(train_images, train_labels)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        end_time = time.time()
        print(f"find_best_kernel completed in {end_time - start_time:.2f} seconds.")

    # finding the best gamma
    def find_best_gamma(self, train_images, train_labels): 
        start_time = time.time()
        parameters = {'gamma':[0.1, 1]}  # Provide gamma values as floating-point numbers
        grid_search = GridSearchCV(SVC(C=10, kernel='rbf'), parameters)
        grid_search.fit(train_images, train_labels)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        end_time = time.time()
        print(f"find_best_gamma completed in {end_time - start_time:.2f} seconds.")

    # it displays support vectors label by label
    def display_support_vectors(self, svm_classifier, train_images, train_labels):
        support_vectors = svm_classifier.support_vectors_
        support_vector_indices = svm_classifier.support_
        support_vector_labels = train_labels[support_vector_indices]
        required_digits = [2, 3, 8, 9]

        for cur_label in required_digits:
            # Filter support vectors for the current digit
            digit_indices = [idx for idx, label in enumerate(support_vector_labels) if label == cur_label]
            digit_sv = support_vectors[digit_indices]

            # plotting support vectors for current label
            fig, axes = plt.subplots(8, 10, figsize=(10, 2))  # 4 column, 20 rows
            fig.suptitle(f'Support Vectors for Label {cur_label}')
            axes = axes.flatten()
            for ax, image in zip(axes, digit_sv):
                ax.imshow(image.reshape(28, 28), cmap='gray')
                ax.axis('off')
            plt.show()

# required labels
required_digits = [2, 3, 8, 9]

is_pca = True
arrange_cvxopt_solver()
solvers.options['show_progress'] = True 
# Load and preprocess data
train_images, train_labels, test_images, test_labels = get_images_labels(required_digits)
print_test_train_labels(train_labels, test_labels)
#train_images, test_images, train_labels, test_labels = make_smaller_sets(train_images, test_images, train_labels, test_labels)
train_images, test_images = reshaping_images_for_svm(train_images, test_images)
train_images,test_images = normalize_images(train_images,test_images)

# step 1.a
LinearQuadraticSolver(train_images,test_images,train_labels,test_labels)
# step 1.b
#ScikitLinearKernel(train_images, test_images, train_labels, test_labels, 1.0, is_feature_extract=False, find_best_params=False)

# step 1.c
#NonLinearQuadraticSolver(train_images, test_images, train_labels, test_labels, is_feature_extract=True)

# step 1.d
#ScikitNonLinearKernel(train_images, test_images, train_labels, test_labels, is_feature_extract=True, display_support_vector=False, find_best_params = True)


