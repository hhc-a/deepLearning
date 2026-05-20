import numpy as np
# TODO 1. import 必要套件/類別
from keras.datasets import mnist
from keras import Model
from keras.layers import Input, Dense

# 指定亂數種子
np.random.seed(7)
# 載入資料集
(X_train, _), (X_test, _) = mnist.load_data()

# TODO 2. 將特徵資料轉換成28*28 = 784的向量
X_train = X_train.reshape(X_train.shape[0], 28*28).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28*28).astype("float32")

# TODO . 執行正規化, 從 0-255 至 0-1
X_train = X_train / 255
X_test = X_test / 255

# TODO 3. 定義 autoencoder 模型
input_img = Input(shape=(784,))
x = Dense(128, activation="relu")(input_img)
encoded = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(x)

# TODO 4. 定義 autoencoder Model
autoencoder = Model(input_img, decoded)
autoencoder.summary()   # 顯示模型摘要資訊
print("--------------------------")

# TODO 5. 定義 encoder 模型
encoder = Model(input_img, encoded)
encoder.summary()    # 顯示模型摘要資訊
print("--------------------------")

# TODO 6. 定義 decoder 模型
decoder_input = Input(shape=(64,))
decoder_layer1 = autoencoder.layers[-2](decoder_input)
decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
decoder = Model(decoder_input, decoder_layer2)
decoder.summary()   # 顯示模型摘要資訊
print("--------------------------")

# 編譯模型
autoencoder.compile(loss="binary_crossentropy", optimizer="adam")

# TODO 7. 訓練模型
autoencoder.fit(X_train, X_train, 
                validation_data=(X_test, X_test), 
                epochs=10, batch_size=256, shuffle=True, verbose=2)

# TODO 8. 準備測試成果
# TODO 8-1. 先用 encoder 壓縮圖片
encoded_imgs = encoder.predict(X_test)
# TODO 8-2. 在用 decoder 解壓縮圖片
decoded_imgs = decoder.predict(encoded_imgs)


# TODO 9. 顯示雜訊圖片, 壓縮圖片和還原圖片
import matplotlib.pyplot as plt
#
n = 10  # 顯示幾個數字
plt.figure(figsize=(20, 8))
for i in range(n):
    # TODO 9-1. 顯示原圖片
    ax = plt.subplot(3, n, i + 1)
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
    # TODO 9-2. 顯示壓縮圖片
    ax = plt.subplot(3, n, i + 1 + n)
    ax.imshow(encoded_imgs[i].reshape(8, 8).T, cmap="gray")
    ax.axis("off")
    # TODO 9-3. 顯示還原圖片
    ax = plt.subplot(3, n, i + 1 + 2*n)
    ax.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
plt.show()