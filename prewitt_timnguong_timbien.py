import cv2
import numpy as np

# Đọc và làm mượt ảnh
img = cv2.imread("4.2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Hàm tích chập
def Tichchap(img, kernel):
    h, w = img.shape
    padded = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + 3, j:j + 3]
            result[i, j] = np.sum(region * kernel)
    return result

# Hàm Prewitt
def Prewitt(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    gx = Tichchap(img, kernel_x)
    gy = Tichchap(img, kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

# Tìm biên Prewitt
edge = Prewitt(blurred)

# Tự động tìm ngưỡng nhị phân bằng Otsu
_, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Hiển thị ảnh
cv2.imshow("Anh ban dau", gray)
cv2.imshow("Tim bien (Prewitt)", edge)
cv2.imshow("Tim nguong (Otsu)", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
