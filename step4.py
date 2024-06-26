from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import idx2numpy
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt


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

# it normalizes the images, simplifies the calculations and execute faster
def normalize_images(train_images, test_images):
    return train_images / 255.0, test_images / 255.0


def train_and_evaluate_svm(train_images, train_labels, test_images, test_labels):
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
def apply_pca(train_images, test_images, n_components=50):
    start_time = time.time()
    pca = PCA(n_components=n_components)
    pca.fit(train_images)
    train_images_pca = pca.transform(train_images)
    test_images_pca = pca.transform(test_images)
    end_time = time.time()
    print(f"PCA Process completed in {end_time - start_time:.2f} seconds.")
    return train_images_pca, test_images_pca


# finding the best C
def find_best_hyperparam(train_images, train_labels): 
    parameters = {'kernel':('poly', 'rbf'), 'C':[0.1, 1,10]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), parameters)
    grid_search.fit(train_images, train_labels)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# finding the best kernel method
def find_best_kernel(train_images, train_labels): 
    parameters = {'kernel':('poly', 'rbf', 'sigmoid')}
    grid_search = GridSearchCV(SVC(C=10), parameters)
    grid_search.fit(train_images, train_labels)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# finding the best gamma
def find_best_gamma(train_images, train_labels): 
    parameters = {'gamma':('0.1', '1', '10')}
    grid_search = GridSearchCV(SVC(C=10, kernel='rbf'), parameters)
    grid_search.fit(train_images, train_labels)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# it displays support vectors label by label
def display_support_vectors(svm_classifier, train_images, train_labels):
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

is_apply_pca = False

# Load and preprocess data
train_images, train_labels, test_images, test_labels = get_images_labels()
train_images, test_images = reshaping_images_for_svm(train_images, test_images)
train_images, test_images = normalize_images(train_images, test_images)
#train_images, test_images, train_labels, test_labels = make_smaller_sets(train_images, test_images, train_labels, test_labels)
if (is_apply_pca):
    train_images, test_images = apply_pca(train_images, test_images)

start_time = time.time()
svm_classifier = train_and_evaluate_svm(train_images, train_labels, test_images, test_labels)
end_time = time.time()
print(f"Process completed in {end_time - start_time:.2f} seconds.")
# if pca is applied, we cannot display
if (not is_apply_pca):
    display_support_vectors(svm_classifier,train_images,train_labels)


# display support vectors
#display_images(support_vectors, support_vector_labels)

#find_best_hyperparam(train_images, train_labels)
#find_best_kernel(train_images, train_labels)
#end_time = time.time()
#print(f"Process completed in {end_time - start_time:.2f} seconds.")
