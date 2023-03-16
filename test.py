import os
import cv2
import numpy as np

# Đường dẫn đến folder ảnh
img_folder = 'dauvao'

# Đường dẫn đến folder mới để lưu ảnh đã được tiền xử lý
processed_folder = 'daura'


# Tạo folder mới nếu chưa tồn tại
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Đọc các file ảnh trong folder
for file_name in os.listdir(img_folder):
    if file_name.endswith('.jpg'):
        # Đường dẫn đến ảnh
        img_path = os.path.join(img_folder, file_name)

        # Tiền xử lý ảnh
        processed_image = preprocess_image(img_path, target_size=(128, 128))

        # Lưu ảnh đã được tiền xử lý vào folder mới
        processed_path = os.path.join(processed_folder, file_name)
        cv2.imwrite(processed_path, processed_image)

def preprocess_image(image_path, target_size):
    # Load the image in RGB format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Normalize the pixel values to [0, 1]
    image = np.array(image) / 255.0
    # Reshape the image into a 4-dimensional tensor
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image
#chưa thấy ai chỉnh sửa
