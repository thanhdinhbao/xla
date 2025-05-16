import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from scipy.spatial.distance import cdist

# Đọc ảnh và chuyển đổi màu
img = cv2.imread('girl.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)

# Chuyển ảnh sang dạng 2D
X = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

# Hàm từ scratch
def Kmean_from_scratch(X, K: int):
    def kmeans_init_centers(X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def kmeans_assign_labels(X, centers):
        D = cdist(X, centers)
        return np.argmin(D, axis=1)

    def kmeans_update_centers(X, labels, K):
        centers = np.zeros((K, X.shape[1]))
        for k in range(K):
            Xk = X[labels == k]
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    def has_converged(centers, new_centers):
        return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0

    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1

    return centers[-1], labels[-1], it
def Kmean(X, K_list: list):
    fig, ax = plt.subplots(3, len(K_list), figsize=(8 * len(K_list), 20))

    for i, K in enumerate(K_list):
        img_based_average = np.zeros_like(X)
        img_based_cluster = np.zeros_like(X)
        img_based_average_from_scratch = np.zeros_like(X)

        # Sklearn KMeans
        kmeans = KMeans(n_clusters=K).fit(X)
        label = kmeans.predict(X)

        # KMeans từ scratch
        clusters, labels_scratch, _ = Kmean_from_scratch(X, K)

        # Gán màu trung tâm
        for k in range(K):
            img_based_cluster[label == k] = kmeans.cluster_centers_[k]
            img_based_average[label == k] = X[label == k].mean(axis=0)
            img_based_average_from_scratch[labels_scratch == k] = X[labels_scratch == k].mean(axis=0)

        # Chuyển về lại dạng ảnh gốc
        reshape = lambda X: X.reshape(img.shape[0], img.shape[1], img.shape[2])
        img1 = reshape(img_based_cluster)
        img2 = reshape(img_based_average)
        img3 = reshape(img_based_average_from_scratch)

        # Hiển thị ảnh
        ax[0][i].imshow(img1, interpolation='nearest')
        ax[0][i].set_title(f'image based cluster, K={K}')
        ax[0][i].axis('off')

        ax[1][i].imshow(img2, interpolation='nearest')
        ax[1][i].set_title(f'image based average, K={K}')
        ax[1][i].axis('off')

        ax[2][i].imshow(img3, interpolation='nearest')
        ax[2][i].set_title(f'image based average by k-mean from scratch, K={K}')
        ax[2][i].axis('off')

    plt.show()
Kmean(X, [2, 5, 7])
