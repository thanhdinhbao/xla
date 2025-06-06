import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_hessian_response(gray, sigma=1.2):
    # Tính đạo hàm bậc hai theo x, y, xy bằng Sobel
    Lxx = cv2.GaussianBlur(cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=5), (3, 3), sigma)
    Lyy = cv2.GaussianBlur(cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=5), (3, 3), sigma)
    Lxy = cv2.GaussianBlur(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5), (3, 3), sigma)

    # Tính định thức Hessian (Det(H) = Lxx*Lyy - Lxy^2)
    det_H = Lxx * Lyy - Lxy ** 2
    return det_H

def detect_keypoints(hessian, threshold=1e-4):
    # Lọc điểm mạnh theo ngưỡng
    keypoints = np.where(hessian > threshold * np.max(hessian))
    return list(zip(keypoints[1], keypoints[0]))  # trả về (x, y)

# Load ảnh và chuyển grayscale
image = cv2.imread("binary.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32) / 255.0

# Tính phản hồi Hessian
hessian_response = compute_hessian_response(gray)

# Lọc các keypoint có phản hồi cao
keypoints = detect_keypoints(hessian_response, threshold=1e-4)

# Vẽ keypoints lên ảnh
for x, y in keypoints:
    cv2.circle(image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

# Hiển thị
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("SURF-like Keypoints (Manual)")
plt.axis('off')
plt.show()
