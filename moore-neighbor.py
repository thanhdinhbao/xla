import numpy as np
import cv2
import matplotlib.pyplot as plt

# Moore neighborhood directions (clockwise)
# [dy, dx] theo thứ tự: trái, trên-trái, trên, trên-phải, phải, dưới-phải, dưới, dưới-trái
directions = [ (0, -1), (-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0), (1, -1)]

def moore_neighbor_trace(binary_img):
    h, w = binary_img.shape
    contour = []

    # Tìm điểm bắt đầu: pixel đầu tiên có giá trị 255
    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 255:
                start = (y, x)
                break
        else:
            continue
        break
    else:
        return []

    current = start
    prev_dir = 7  # bắt đầu kiểm tra từ bên trái

    while True:
        contour.append(current)
        found = False
        for i in range(8):
            dir_idx = (prev_dir + i) % 8
            dy, dx = directions[dir_idx]
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < h and 0 <= nx < w and binary_img[ny, nx] == 255:
                current = (ny, nx)
                prev_dir = (dir_idx + 5) % 8  # quay lại 3 bước để kiểm tiếp
                found = True
                break
        if not found or current == start:
            break

    return contour

# Đọc ảnh nhị phân (phải là ảnh trắng đen)
image = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Phát hiện contour bằng Moore-Neighbor
contour = moore_neighbor_trace(binary)

# Hiển thị kết quả
for y, x in contour:
    binary[y, x] = 127  # tô màu đường viền

plt.imshow(binary, cmap='gray')
plt.title("Contour Detected (Moore Neighbor)")
plt.axis('off')
plt.show()
