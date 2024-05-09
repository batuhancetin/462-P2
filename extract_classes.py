import idx2numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import time


#need to install 
#pip3 install numpy idx2numpy --break-system-packages
#pip3 install scikit-learn --break-system-packages


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
    train_images = train_images[:10000]
    test_images = test_images[:1000]
    train_labels = train_labels[:10000]
    test_labels = test_labels[:1000]
    return train_images, test_images, train_labels, test_labels

# it starts linar svm with given regularization parameters and image & label datasets 
def start_svm(regulirazation_param, train_images, train_labels, test_images, test_labels):
    print("Regularization Param (C): " + str(regulirazation_param))
    start_time = time.time()
    # Initialize the LinearSVC model with default parameters to start
    # dual=False when n_samples > n_features
    svm_model = LinearSVC(C=regulirazation_param,dual=False)  
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
def find_best_param():
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


# it trains scikit-learn’s soft margin primal SVM function with linear kernel. Its default regularization param is 1.0 to make it soft-margin
def step_b(train_images, train_labels, test_images, test_labels):
    regulirazation_param = 1.0 
    start_svm(regulirazation_param, train_images, train_labels, test_images, test_labels)
    #find_best_param()

train_images, train_labels, test_images, test_labels = get_images_labels()
print_test_train_labels(train_labels, test_labels)
#train_images, test_images, train_labels, test_labels = make_smaller_sets(train_images, test_images, train_labels, test_labels)
train_images, test_images = reshaping_images_for_svm(train_images, test_images)
step_b(train_images, train_labels, test_images, test_labels)
#find_best_param()


