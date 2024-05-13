import numpy as np
from collections import Counter
from keras.datasets import mnist
from sklearn.decomposition import PCA


# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Select only digits 2, 3, 8, and 9
selected_digits = [2, 3, 8, 9]
selected_indices = np.isin(train_y, selected_digits)
x_selected = train_X[selected_indices]
y_selected = train_y[selected_indices]

# Normalize the data
x_normalized = x_selected / 255.0

# Flatten the images
x_flattened = x_normalized.reshape(x_normalized.shape[0], -1)


# Implement k-means from scratch
def kmeans_with_euclidian(x, no_of_clusters, max_iters=100):
    # Randomly initialize centroids
    centroids = x[np.random.choice(len(x), no_of_clusters, replace=False)]
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # Update centroids
        new_centroids = np.zeros((no_of_clusters, x.shape[1]))
        for i in range(no_of_clusters):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


def kmeans_with_cosine_similarity(x, no_of_clusters, max_iters=100):
    # Randomly initialize centroids
    centroids = x[np.random.choice(len(x), no_of_clusters, replace=False)]
    for _ in range(max_iters):
        # Calculate cosine similarity
        similarities = np.dot(x, centroids.T) / (
                    np.linalg.norm(x, axis=1)[:, np.newaxis] * np.linalg.norm(centroids, axis=1))
        # Assign each data point to the nearest centroid
        labels = np.argmax(similarities, axis=1)
        # Update centroids
        new_centroids = np.zeros((no_of_clusters, x.shape[1]))
        for i in range(no_of_clusters):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


def assign_labels_to_clusters(labels, true_labels, k):
    assigned_labels = {}
    for cluster_idx in range(k):
        cluster_labels = true_labels[labels == cluster_idx]
        # Count occurrences of each label in the cluster
        label_counts = Counter(cluster_labels)
        # Get the most common label
        most_common_label = label_counts.most_common(1)[0][0]
        assigned_labels[cluster_idx] = most_common_label
    return assigned_labels


def calculate_accuracy(true_labels, predicted_labels, k):
    correct = 0
    assigned_labels = assign_labels_to_clusters(predicted_labels, true_labels, k)
    for i in range(true_labels.shape[0]):
        if assigned_labels[predicted_labels[i]] == true_labels[i]:
            correct += 1
    return correct / true_labels.shape[0]


def calculate_sse(data, centroids, labels):
    sse = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            cluster_sse = np.sum((cluster_points - centroid) ** 2)
            sse += cluster_sse
    return sse


# this function extracts the features, it defaults extract features until 50 features remaining (from 786 28*28)
def apply_pca(train_images, n_components=50):
    pca = PCA(n_components=n_components)
    pca.fit(train_images)
    train_images_pca = pca.transform(train_images)
    return train_images_pca


# Perform k-means clustering on selected digits
k = len(selected_digits)
# Perform k-means clustering with Euclidian distance
centroids_euclidian, labels_euclidian = kmeans_with_euclidian(x_flattened, k)

accuracy_euclidian = calculate_accuracy(y_selected, labels_euclidian, k)
sse_euclidian = calculate_sse(x_flattened, centroids_euclidian, labels_euclidian)

# Perform k-means clustering with cosine similarity
centroids_cosine, labels_cosine = kmeans_with_cosine_similarity(x_flattened, k)

accuracy_cosine = calculate_accuracy(y_selected, labels_cosine, k)
sse_cosine = calculate_sse(x_flattened, centroids_cosine, labels_cosine)

# Perform k-means clustering on extracted features with Euclidian distance
extracted_images = apply_pca(train_images=x_flattened)
centroids_euclidian_extracted, labels_euclidian_extracted = kmeans_with_euclidian(extracted_images, k)

accuracy_euclidian_extracted = calculate_accuracy(y_selected, labels_euclidian_extracted, k)
sse_euclidian_extracted = calculate_sse(extracted_images, centroids_euclidian_extracted, labels_euclidian_extracted)


print("Accuracy of euclidian: ", accuracy_euclidian)
print("SSE of euclidian: ", sse_euclidian)

print("Accuracy of cosine similarity: ", accuracy_cosine)
print("SSE of cosine similarity: ", sse_cosine)

print("Accuracy of euclidian features extracted ", accuracy_euclidian_extracted)
print("SSE of euclidian features extracted ", sse_euclidian_extracted)

