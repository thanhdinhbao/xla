import cv2

# Đọc ảnh gốc ở dạng grayscale
image = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)

# Áp dụng Otsu's thresholding
# Tham số 0 là threshold ban đầu (không dùng), Otsu sẽ tự tính ngưỡng tối ưu
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Hiển thị ảnh kết quả
cv2.imshow("Original", image)
cv2.imshow("Otsu Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
