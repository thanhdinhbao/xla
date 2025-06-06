import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread("your_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # chuyển sang RGB để hiển thị bằng matplotlib

# Áp dụng Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.5)

# Hiển thị kết quả
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Gaussian Blurred")
plt.imshow(blurred)
plt.axis("off")

plt.show()
