import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang grayscale
image = cv2.imread('binary.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(float) / 255.0  # Chuẩn hóa

# Phát hiện đốm bằng Laplacian of Gaussian (LoG)
blobs = blob_log(gray, max_sigma=30, num_sigma=10, threshold=0.02)

# Tính bán kính tương ứng (r ≈ sqrt(2) * sigma)
blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

# Hiển thị kết quả
fig, ax = plt.subplots()
ax.imshow(gray, cmap='gray')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=1.5, fill=False)
    ax.add_patch(c)

plt.title("Đặc trưng đốm bằng LoG")
plt.axis('off')
plt.show()
