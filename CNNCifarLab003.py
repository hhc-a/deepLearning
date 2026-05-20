import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow / Keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization
)
from keras.utils import to_categorical

# 固定亂數種子
np.random.seed(10)

# =========================
# 1. 載入 CIFAR10 資料集
# =========================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# =========================
# 2. 正規化
# 將 0~255 轉成 0~1
# =========================
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# =========================
# 3. One-hot Encoding
# =========================
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =========================
# 4. 建立 CNN 模型
# =========================
model = Sequential()

# 輸入層
model.add(Input(shape=X_train.shape[1:]))

# =========================
# Block 1
# =========================
model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(BatchNormalization())

model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation="relu"
))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# =========================
# Block 2
# =========================
model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(BatchNormalization())

model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    activation="relu"
))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# =========================
# Block 3
# =========================
model.add(Conv2D(
    128,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))

# =========================
# 全連接層
# =========================
model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.3))

# 輸出層
model.add(Dense(10, activation="softmax"))

# 顯示模型摘要
model.summary()

print("-----------------------------")

# =========================
# 5. 編譯模型
# =========================
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# =========================
# 6. 訓練模型
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    verbose=2
)

# =========================
# 7. 評估模型
# =========================
print("\nTesting ...")

# 訓練資料準確率
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集準確度 = {:.2f}%".format(accuracy * 100))

# 測試資料準確率
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集準確度 = {:.2f}%".format(accuracy * 100))

# =========================
# 8. 儲存模型
# =========================
print("\nSaving Model : cifar10_improved.keras")
model.save("cifar10_improved.keras")

# =========================
# 9. 畫 Loss 圖
# =========================
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    loss,
    "bo-",
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    "ro--",
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()

# =========================
# 10. 畫 Accuracy 圖
# =========================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    acc,
    "bo-",
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_acc,
    "ro--",
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()

# =========================
# 11. 預測與混淆矩陣
# =========================

# 重新取得原始 y_test
(_, _), (_, y_test_original) = cifar10.load_data()

print("\nPredicting ...")

# 預測
y_pred = model.predict(X_test)

# 取最大機率類別
y_pred_result = np.argmax(y_pred, axis=1)

# 建立混淆矩陣
tb = pd.crosstab(
    y_test_original.astype(int).flatten(),
    y_pred_result.astype(int),
    rownames=["Label"],
    colnames=["Predict"]
)

print("\nConfusion Matrix")
print(tb)

# 輸出 HTML
tb.to_html("cnn_confusion_matrix.html")

print("\nConfusion matrix saved : cnn_confusion_matrix.html")