import cv2
import numpy as np
import tensorflow as tf

# Đọc ảnh và chuẩn hóa giá trị các điểm ảnh về khoảng từ 0 đến 1
img = cv2.imread('test5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype('float32') / 255.0

# Xây dựng mạng neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile mô hình
model.compile(optimizer='sgd', loss='binary_crossentropy')

# Huấn luyện mô hình
model.fit(img.reshape(-1, img.shape[0] * img.shape[1]), np.array([1]), epochs=100)

# Dự đoán giá trị đầu ra của ảnh
output = model.predict(img.reshape(-1, img.shape[0] * img.shape[1]))
print("ketqua"+str(output))
# # Chuyển đổi giá trị từ khoảng (0, 1) sang khoảng (0, 255)
# output = (output.reshape(img.shape) * 255.0)
# output = output.astype('uint8')

# # Hiển thị ảnh sau khi xử lý
# cv2.imshow('Output', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
