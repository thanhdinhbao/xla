import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Hàm tính histogram 3 kênh màu
def calcHist(image, bins=64, ranges=[0, 256]):
    hist = np.zeros((bins * 3), dtype=np.float32)
    pixel_count = 0
    b, g, r = cv2.split(image)
    for channel in (b, g, r):
        channel_hist, _ = np.histogram(channel, bins=bins, range=ranges)
        hist[pixel_count : pixel_count + bins] = channel_hist
        pixel_count += bins
    return hist

# Đọc ảnh test và resize
test_img = cv2.imread('XLAandTGMT\\Buoi6\\test\\test\\t2.jpg')
test_img = cv2.resize(test_img, (300, 300))

# Tạo danh sách ảnh dữ liệu và histogram
data_imgs = []
data_hists = []

# Load ảnh nhóm a
for i in range(1, 12):
    data_img = cv2.imread('XLAandTGMT\\Buoi6\\data\\data\\a{}.jpg'.format(i))
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img).flatten()
    data_imgs.append(data_img)
    data_hists.append(hist)

# Load thêm ảnh nhóm c
for i in range(1, 6):
    data_img = cv2.imread('XLAandTGMT\\Buoi6\\data\\data\\c{}.jpg'.format(i))
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img).flatten()
    data_imgs.append(data_img)
    data_hists.append(hist)

# Histogram của ảnh test
test_hist = calcHist(test_img).flatten()
# Số lượng ảnh gần nhất
K = 3

# Khởi tạo và fit KNN
neigh = NearestNeighbors(n_neighbors=K)
neigh.fit(data_hists)

# Tìm các ảnh gần nhất
distances, indices = neigh.kneighbors([test_hist])

# Hiển thị ảnh test
cv2.imshow('Test image', test_img)

# Hiển thị các ảnh gần nhất
for i in range(K):
    data_img = data_imgs[indices[0][i]]
    cv2.imshow('Data image {}'.format(i+1), data_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
