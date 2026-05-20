import numpy as np
# TODO 1. import 必要套件
from keras.datasets import mnist
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# 指定亂數種子
np.random.seed(7)

# 載入資料集
(X_train, _), (X_test, _) = mnist.load_data()
# TODO 2. 轉換成 4D 張量
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

# TODO 3. 執行正規化, 從 0-255 轉換至 0-1
X_train = X_train / 255
X_test = X_test / 255

# 定義 autoencoder 模型
input_img = Input(shape=(28,28,1))
x = Conv2D(16, (3,3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2,2), padding="same")(x)
x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2), padding="same")(x)
x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2,2), padding="same")(x)
x = Conv2D(8, (3,3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation="relu")(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# TODO 4. 定義 autoencoder Model
autoencoder = Model(input_img, decoded)
autoencoder.summary()   # 顯示模型摘要資訊
print("--------------------------")
# TODO 5. 定義 encoder 模型
encoder = Model(input_img, encoded)
encoder.summary()    # 顯示模型摘要資訊
print("--------------------------")
# TODO 6. 定義 decoder 模型
decoder_input = Input(shape=(4,4,8))
decoder_layer = autoencoder.layers[-7](decoder_input)
decoder_layer = autoencoder.layers[-6](decoder_layer)
decoder_layer = autoencoder.layers[-5](decoder_layer)
decoder_layer = autoencoder.layers[-4](decoder_layer)
decoder_layer = autoencoder.layers[-3](decoder_layer)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(decoder_input, decoder_layer)
decoder.summary()   # 顯示模型摘要資訊

# 編譯模型
autoencoder.compile(loss="binary_crossentropy", optimizer="adam")
# 訓練模型
autoencoder.fit(X_train, X_train, 
                validation_data=(X_test, X_test), 
                epochs=10, batch_size=128, shuffle=True, verbose=2)

# TODO 7. 壓縮圖片
encoded_imgs = encoder.predict(X_test)
# TODO 8.解壓縮圖片
decoded_imgs = decoder.predict(encoded_imgs)
# 顯示雜訊圖片, 壓縮圖片和還原圖片

# TODO 9.顯示結果
import matplotlib.pyplot as plt
#
n = 10  # 顯示幾個數字
plt.figure(figsize=(20, 8))
for i in range(n):
    # TODO 9-1. 顯示原來圖片
    ax = plt.subplot(3, n, i + 1)
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
    # TODO 9-2 顯示壓縮圖片
    ax = plt.subplot(3, n, i + 1 + n)
    ax.imshow(encoded_imgs[i].reshape(4, 4*8).T, cmap="gray")
    ax.axis("off")
    # TODO 9-3. 顯示還原圖片
    ax = plt.subplot(3, n, i + 1 + 2*n)
    ax.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
plt.show()
