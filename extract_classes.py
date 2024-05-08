import idx2numpy
import numpy as np

#need to install 
#pip3 install numpy idx2numpy --break-system-packages
#pip3 install scikit-learn --break-system-packages


# Paths to the dataset files
file_train_images = 'mnist/train-images-idx3-ubyte'
file_train_labels = 'mnist/train-labels-idx1-ubyte'
file_test_images = 'mnist/t10k-images-idx3-ubyte'
file_test_labels = 'mnist/t10k-labels-idx1-ubyte'

# Load the dataset
train_images = idx2numpy.convert_from_file(file_train_images)
train_labels = idx2numpy.convert_from_file(file_train_labels)
test_images = idx2numpy.convert_from_file(file_test_images)
test_labels = idx2numpy.convert_from_file(file_test_labels)

# Get the Required Digits from Labels
required_digits = [2, 3, 8, 9]

# it filters images w.r.t their labels
# it first only the labels consists required digits then return image and label list correspondingly
def filter_digits(images, labels, digits):
    mask = np.isin(labels, digits)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    return filtered_images, filtered_labels

# Apply the filtering for required images
train_images, train_labels = filter_digits(train_images, train_labels, required_digits)
test_images, test_labels = filter_digits(test_images, test_labels, required_digits)

# reshaping them 28x28 to 786-dimensional vector
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
