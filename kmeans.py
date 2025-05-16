import numpy as np
import cv2
from sklearn.cluster import KMeans

# Đọc ảnh
img = cv2.imread('girl.png')
cv2.imshow("Original Img", img)

# Chuyển ảnh về dạng 2D (flatten) để huấn luyện KMeans
X = img.reshape((-1, 3))  # (số pixel, 3 kênh màu)
K = 30  # Số cụm màu

# Áp dụng KMeans
kmeans = KMeans(n_clusters=K).fit(X)
label = kmeans.predict(X)

# Gán màu trung tâm cụm cho mỗi pixel
img4 = np.zeros_like(X)
for i in range(K):
    img4[label == i] = kmeans.cluster_centers_[i]

# Đưa ảnh về lại shape gốc
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
cv2.imshow("New Img", img5.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
