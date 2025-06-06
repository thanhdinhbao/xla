import cv2

# Đọc ảnh gốc và chuyển sang grayscale
image = cv2.imread("your_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng Gaussian Blur để giảm nhiễu
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

# Phát hiện biên bằng Canny (thường chọn 2 ngưỡng: thấp và cao)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Hiển thị kết quả
cv2.imshow("Original", image)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
