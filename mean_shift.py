import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread("girl.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Áp dụng Mean Shift Filtering
segmented = cv2.pyrMeanShiftFiltering(image, sp=15, sr=30)

# Hiển thị kết quả
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Ảnh sau Mean Shift")
plt.imshow(segmented)
plt.axis('off')

plt.show()
