import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical

# 指定亂數種子
np.random.seed(10)
# 載入資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test.astype("float32") / 255
# One-hot編碼
y_test_bk = y_test.copy()   # 備份 Y_test 資料集
y_test = to_categorical(y_test)

# TODO 1. 載入Keras模型
model = load_model("cifar10.keras")

# 計算分類的預測值
print("\nPredicting ...")
y_pred = model.predict(X_test)
y_pred_result = np.argmax(y_pred, axis=1)
# TODO 2. 顯示混淆矩陣
tb = pd.crosstab(y_test_bk.astype(int).flatten(), 
                 y_pred_result.astype(int),
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("ch9-1-3.html")
