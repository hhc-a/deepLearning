import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.models import load_model

np.random.seed(7)  # 指定亂數種子
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}

# 載入資料集
df = pd.read_csv("./iris_data.csv")
df["target"] = df["target"].map(target_mapping)
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,0:4].astype(float)
y = to_categorical(dataset[:,4])
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割成訓練和測試資料集
X_train, y_train = X[:120], y[:120]     # 訓練資料前120筆
X_test, y_test = X[120:], y[120:]       # 測試資料後30筆

#
# (A) 訓練

#建立Keras的Sequential模型
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
print("Training ...")
model.fit(X_train, y_train, epochs=100, batch_size=5)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("準確度 = {:.2f}".format(accuracy))

# TODO 1. 儲存Keras模型
print("Saving Model: iris_0429.keras ...")
model.save("iris_0429.keras")



# (B) 使用 model

# TODO 2. 載入Keras模型
model = load_model("iris_0429.keras")

# 計算分類的預測值
print("\nPredicting ...")
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
y_target = dataset[:,4][120:].astype(int)
print(y_target)

# TODO 3. 顯示混淆矩陣
tb = pd.crosstab(y_target, y_pred, rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("ch6-1-3.html")
