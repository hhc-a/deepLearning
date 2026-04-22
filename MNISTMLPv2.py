import numpy as np
# Keras 2.x
#from tensorflow import keras
#from tensorflow.keras.datasets import mnist
# Keras 3.x
import keras

# TODO 1. import mnist
from keras.datasets import mnist 
#
import matplotlib.pyplot as plt

# TODO 2. 載入 MNIST 手寫數字資料集，回傳訓練集與測試集的影像和標籤
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# TODO 4.
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# 利用 List 參數方式建立模型
model = keras.Sequential([
    # 將 28x28 的二維影像攤平成 784 維向量，方便送入全連接層
    keras.layers.Flatten(input_shape=(28, 28)),
    # 隱藏層：128 個神經元，使用 ReLU 增加非線性表達能力
    keras.layers.Dense(128, activation='relu'),
    # 輸出層：10 個類別，使用 softmax 輸出每個數字的機率
    keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
# adam: 常用最佳化器；
# sparse_categorical_crossentropy: 適合整數類別標籤
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 顯示模型各層結構與參數數量
model.summary()

# 訓練模型
print("\n◆Train:")
# 使用訓練資料反覆學習 10 次（10 個 epochs）
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
print("Done")

# 預測
print("\n◆Predict:")
# 對測試資料做分類預測，結果為每張圖屬於 10 類的機率分布
predictions = model.predict(test_images, verbose=1)
print(f"Predictions for the first 5 test images:\n{predictions[:5]}")

# 正確率
print("\n◆Evaluate:")
# 在測試集上計算 loss 與 accuracy，檢查模型泛化能力
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test accuracy:{100 * test_acc:5.2f}%")

# 驗證單一個數字
num = 2
# 從第 num 張測試圖片的 10 類機率中，取出機率最大的類別索引
result = np.argmax(predictions[num])
print(f"The result of image {num} is [{result}].")

# 顯示辨識結果圖片
num_display = 10
# TODO 3. 建立 2x5 子圖，一次顯示前 10 張測試圖片與預測結果
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("MNIST Prediction Results", fontsize=14)
for i, ax in enumerate(axes.flat):
    # 顯示灰階手寫數字影像
    ax.imshow(test_images[i], cmap='gray')
    # 取出模型預測類別與真實標籤，並用顏色區分是否預測正確
    predicted = np.argmax(predictions[i])
    actual = test_labels[i]
    color = 'green' if predicted == actual else 'red'
    ax.set_title(f"Pred: {predicted}\nActual: {actual}", color=color, fontsize=10)
    ax.axis('off')
# 自動調整版面，避免標題或子圖重疊
plt.tight_layout()
plt.show()

# ============================================================
# 視覺化訓練過程：Accuracy
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 視覺化訓練過程：Loss
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
