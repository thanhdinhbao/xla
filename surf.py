import cv2
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang grayscale
image = cv2.imread("girl.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Khởi tạo đối tượng SURF (threshold càng lớn thì ít keypoint hơn)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# Phát hiện keypoints và tính descriptor
keypoints, descriptors = surf.detectAndCompute(gray, None)

# Vẽ keypoints lên ảnh
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Hiển thị
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SURF Keypoints")
plt.axis('off')
plt.show()
