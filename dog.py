import cv2
import numpy as np
from skimage.feature import blob_dog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang grayscale
image = cv2.imread("your_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Chuyển về kiểu float và scale về [0, 1] để phù hợp với scikit-image
gray_float = gray.astype(float) / 255.0

# Phát hiện đốm bằng phương pháp DoG
blobs = blob_dog(gray_float, max_sigma=30, threshold=0.1)

# DoG trả về (y, x, sigma); bán kính ≈ sqrt(2) * sigma
blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

# Hiển thị kết quả
fig, ax = plt.subplots()
ax.imshow(gray, cmap='gray')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
    ax.add_patch(c)

plt.title("Đặc trưng đốm bằng DoG")
plt.axis('off')
plt.show()
