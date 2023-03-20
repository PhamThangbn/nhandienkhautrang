from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2
import numpy as np
import glob
DIRECTORY = r"D:/Downloads/demo neural/dataset"
CATEGORIES = ["with_mask", "without_mask"]



# Tải dữ liệu hình ảnh và nhãn
print("[INFO] Đang tải hình ảnh...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    files = glob.glob(os.path.join(path, "*.jpg"))
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, (250, 250))
        # image = cv2.imread(img_path)
        # if image is not None:
        #     image = cv2.resize(image, (250, 250))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        data.append(image)
        labels.append(category)


# Chuyển nhãn sang dạng nhị phân
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Chia thành tập huấn luyện và kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#Chuẩn hóa dữ liệu
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX.reshape(trainX.shape[0], -1))
testX = scaler.transform(testX.reshape(testX.shape[0], -1))

# Tạo mô hình MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, alpha=1e-4, solver='adam', random_state=42, verbose=10, learning_rate_init=1e-4)

# Train model
print("Mô hình đào tạo...")
model.fit(trainX, np.argmax(trainY, axis=1))

# Dự đoán trên tập kiểm tra
print("Đánh giá mạng...")
preds = model.predict(testX)

# # Hiển thị báo cáo phân loại
# print(classification_report(np.argmax(testY, axis=1), preds, target_names=lb.classes_))

# Tạo model và train model ở đây

# # Lưu model
# filename = 'mask_detectormd.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump(model, f)
print("ketqua" + str (preds))
print("ketqua" + str (trainX[3]))
# print("ketqua" + str (labels))