'''
原本的程式使用 Flatten → Dense → Dense，
也就是把 28×28 影像直接攤平成 784 維向量，再交給全連接層學習；
這會失去影像的空間結構，因此辨識率容易停滯。
'''

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras import layers

# ============================================================
# 1. 載入 MNIST 手寫數字資料集
# ============================================================
# train_images: 訓練影像，形狀為 (60000, 28, 28)
# train_labels: 訓練標籤，數字 0~9
# test_images: 測試影像，形狀為 (10000, 28, 28)
# test_labels: 測試標籤，數字 0~9
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# ============================================================
# 2. 資料前處理
# ============================================================

# 將像素值由 0~255 正規化為 0~1
# 這可以讓模型訓練更穩定、收斂更快
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# CNN 的 Conv2D 層需要輸入格式為：
# (資料筆數, 高度, 寬度, 通道數)
# MNIST 是灰階影像，所以通道數為 1
#
# TODO 1. 改變 NDaddar 維度
# 原本形狀: (60000, 28, 28) -> 改成形狀: (60000, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

print("Train image shape:", train_images.shape)
print("Test image shape:", test_images.shape)


# TODO 2. - 建立 CNN 模型
# ============================================================
# 3. 建立 CNN 模型
# ============================================================
model = keras.Sequential([
    # --------------------------------------------------------
    # 第一層卷積層
    # filters=32：使用 32 個卷積核學習影像特徵
    # kernel_size=(3, 3)：每次看 3x3 的區域
    # activation='relu'：加入非線性能力
    # input_shape=(28, 28, 1)：輸入為 28x28 灰階影像
    # --------------------------------------------------------
    
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # --------------------------------------------------------
    # 池化層
    # MaxPooling2D 會縮小特徵圖大小
    # 可降低參數量，也能保留重要特徵
    # --------------------------------------------------------
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --------------------------------------------------------
    # 第二層卷積層
    # 使用更多 filters，讓模型學習更複雜的筆畫特徵
    # --------------------------------------------------------
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # --------------------------------------------------------
    # 第三層卷積層
    # 進一步抽取高階特徵，例如數字整體形狀
    # --------------------------------------------------------
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # --------------------------------------------------------
    # 將卷積後的特徵圖攤平成一維向量
    # 才能送入 Dense 全連接層做分類
    # --------------------------------------------------------
    layers.Flatten(),

    # --------------------------------------------------------
    # 全連接層
    # 將 CNN 抽出的影像特徵整合後進行分類判斷
    # --------------------------------------------------------
    layers.Dense(64, activation="relu"),

    # --------------------------------------------------------
    # Dropout 可降低過度擬合
    # 訓練時會隨機關閉部分神經元
    # --------------------------------------------------------
    layers.Dropout(0.3),
    
    
    # --------------------------------------------------------
    # 輸出層
    # MNIST 有 10 類：0~9
    # softmax 會輸出每一類的機率
    # --------------------------------------------------------
    layers.Dense(10, activation="softmax")
])


# ============================================================
# 4. 編譯模型
# ============================================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 顯示模型結構
model.summary()


# ============================================================
# 5. 訓練模型
# ============================================================
print("\n◆ Train:")

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

print("Training Done")


# ============================================================
# 6. 評估模型
# ============================================================
print("\n◆ Evaluate:")

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)

print(f"Test accuracy: {100 * test_acc:.2f}%")


# ============================================================
# 7. 預測測試資料
# ============================================================
print("\n◆ Predict:")

predictions = model.predict(test_images, verbose=1)

print("Predictions for the first 5 test images:")
print(predictions[:5])


# ============================================================
# 8. 驗證單一測試圖片
# ============================================================
num = 2

result = np.argmax(predictions[num])
actual = test_labels[num]

print(f"The result of image {num} is [{result}].")
print(f"The actual label of image {num} is [{actual}].")


# ============================================================
# 9. 顯示前 10 張測試圖片與預測結果
# ============================================================

num_display = 10

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("MNIST CNN Prediction Results", fontsize=14)

for i, ax in enumerate(axes.flat):
    # test_images[i] 目前形狀是 (28, 28, 1)
    # imshow 顯示時可使用 squeeze() 轉回 (28, 28)
    ax.imshow(test_images[i].squeeze(), cmap="gray")

    predicted = np.argmax(predictions[i])
    actual = test_labels[i]

    color = "green" if predicted == actual else "red"

    ax.set_title(
        f"Pred: {predicted}\nActual: {actual}",
        color=color,
        fontsize=10
    )

    ax.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 10. 顯示訓練過程：accuracy 與 loss
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

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# TODO 3. 視覺化 Filters & Features
# ============================================================
# CNN 視覺化整合函式
# ============================================================

def visualize_cnn_filters(model):
    """
    視覺化第一層 Conv2D filters
    """

    first_conv_layer = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            first_conv_layer = layer
            break

    if first_conv_layer is None:
        print("No Conv2D layer found.")
        return

    filters, biases = first_conv_layer.get_weights()

    print("First Conv2D layer:", first_conv_layer.name)
    print("Filter shape:", filters.shape)

    f_min, f_max = filters.min(), filters.max()

    if f_max - f_min != 0:
        filters = (filters - f_min) / (f_max - f_min)

    num_filters = min(32, filters.shape[-1])

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle("First Conv2D Layer Filters", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(filters[:, :, 0, i], cmap="gray")
            ax.set_title(f"Filter {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model, image, max_maps=32):
    """
    視覺化輸入圖片經過 CNN Conv2D layers 後的 feature maps
    """

    # 找出所有 Conv2D 層
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Conv2D)
    ]

    if len(conv_layers) == 0:
        print("No Conv2D layer found.")
        return

    # 建立中間層輸出模型
    activation_model = keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in conv_layers]
    )

    # 加入 batch 維度
    image_batch = np.expand_dims(image, axis=0)

    # 取得每層 feature maps
    activations = activation_model.predict(image_batch)

    # 顯示原圖
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

    # 顯示每一層 Conv2D feature maps
    for layer, activation in zip(conv_layers, activations):
        feature_maps = activation[0]

        num_maps = min(max_maps, feature_maps.shape[-1])

        cols = 8
        rows = int(np.ceil(num_maps / cols))

        plt.figure(figsize=(12, rows * 1.8))
        plt.suptitle(f"Feature Maps - {layer.name}", fontsize=14)

        for i in range(num_maps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(feature_maps[:, :, i], cmap="gray")
            plt.title(f"Map {i}", fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


# ============================================================
# 執行 CNN 視覺化
# ============================================================

# 1. 視覺化第一層 filters
visualize_cnn_filters(model)

# 2. 視覺化某一張測試圖片的 feature maps
img_index = 0
visualize_feature_maps(model, test_images[img_index], max_maps=32)