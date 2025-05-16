import numpy as np
import cv2

def cotrang(gray):
    res = np.zeros((gray.shape[0], gray.shape[1] - 1), dtype='uint8')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] - 1):
            if gray[i][j] == 0 or gray[i][j+1] == 0:
                res[i][j] = 0
            else:
                res[i][j] = 255
    return res

def giantrang(gray):
    res = np.zeros((gray.shape[0], gray.shape[1] - 1), dtype='uint8')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] - 1):
            if gray[i][j] == 255 or gray[i][j+1] == 255:
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res

# Đọc ảnh nhị phân
img = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)  # ảnh đầu vào phải là nhị phân trắng đen (0 và 255)
cv2.imshow("Anh goc", img)

# Áp dụng
co_trang_img = cotrang(img)
gian_trang_img = giantrang(img)

# Hiển thị kết quả
cv2.imshow("Ket qua cotrang", co_trang_img)
cv2.imshow("Ket qua giantrang", gian_trang_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
