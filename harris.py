import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang grayscale
image = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)  # cornerHarris yêu cầu float32

# Áp dụng Harris Corner Detection
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# Dãn vùng biên để dễ nhìn
dst = cv2.dilate(dst, None)

# Đánh dấu các góc trên ảnh gốc
image[dst > 0.01 * dst.max()] = [0, 0, 255]  # tô đỏ vị trí góc

# Hiển thị kết quả
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()
