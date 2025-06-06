import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# Đọc ảnh và chuyển grayscale
image = cv2.imread("your_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32) / 255.0

# B1: Làm mượt ảnh với Gaussian
blur = cv2.GaussianBlur(gray, (9, 9), sigmaX=2.0)

# B2: Áp dụng toán tử Laplacian sau Gaussian
log = cv2.Laplacian(blur, ddepth=cv2.CV_64F, ksize=5)

# B3: Lấy giá trị tuyệt đối để dễ xử lý (vì LoG có thể âm)
log_abs = np.abs(log)

# B4: Áp dụng cực đại cục bộ để tìm điểm có giá trị LoG lớn nhất trong lân cận
local_max = maximum_filter(log_abs, size=10) == log_abs

# Ngưỡng: chỉ giữ lại những điểm thực sự nổi bật
threshold = 0.03
blobs = np.where((log_abs > threshold) & local_max)

# Hiển thị ảnh và vẽ blob
plt.imshow(gray, cmap='gray')
for y, x in zip(blobs[0], blobs[1]):
    plt.plot(x, y, 'ro', markersize=5, markeredgewidth=1)
plt.title("Đặc trưng đốm bằng LoG thủ công")
plt.axis('off')
plt.show()
